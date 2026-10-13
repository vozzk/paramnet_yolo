import os

# 权重缓存在 E:\torch\hub\checkpoints;进程环境变量缺失时兜底,避免 torch.hub 尝试联网下载
os.environ.setdefault("TORCH_HOME", r"E:\torch")

import cv2
import numpy as np
import torch

from perspective2d import PerspectiveFields

DEFAULT_VERSION = "Paramnet-360Cities-edina-centered"


class GravityEstimator:
    """ParamNet 重力场估计(2D 方案,作为 fallback 保留)。

    固定机位下 calibrate() 一次,之后每帧查表复用;
    相机移动/转向/变焦/切换分辨率后必须重新 calibrate()。
    """

    def __init__(self, version=DEFAULT_VERSION, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PerspectiveFields(version).eval().to(self.device)
        # (2, H, W):通道 0=x 分量、1=y 分量(图像坐标,y 向下),逐像素单位向量
        self.field = None

    def infer(self, frame_bgr):
        """单帧推理,返回原始预测 dict(pred_gravity_original/pred_roll/pred_pitch/pred_vfov)。"""
        with torch.no_grad():
            return self.model.inference(img_bgr=frame_bgr)

    def calibrate(self, frames):
        """对同机位的若干帧求平均重力场,压单帧噪声(ParamNet 单帧约 1~2 度)。"""
        fields = [
            self.infer(f)["pred_gravity_original"].float().cpu().numpy() for f in frames
        ]
        self.field = np.mean(fields, axis=0)
        self.field /= np.clip(
            np.linalg.norm(self.field, axis=0, keepdims=True), 1e-6, None
        )
        return self

    def up_at(self, cx, cy):
        """取 (cx, cy) 处的重力向上 2D 单位向量 (dx, dy);dy<0 表示指向图像上方。"""
        if self.field is None:
            raise RuntimeError("请先调用 calibrate()")
        v = self.field[:, int(cy), int(cx)]
        n = float(np.linalg.norm(v))
        if n <= 1e-6:
            return np.array([0.0, -1.0], dtype=np.float32)
        return v / n


MOGE_WEIGHTS = r"E:\modelscope\hub\models\moge-2-vitl\model.pt"


class MoGeGravity:
    """MoGe 点云重力(3D 方案,主管线):地面平面拟合出重力方向 + 瓶身点云 PCA 出轴向。

    相对 ParamNet 的优势:轴向与重力同源(同一份点云),且天然免疫
    "径向躺倒"盲区(2D 投影夹角法对顺视线方向躺倒的瓶子必然漏判)。
    固定机位下重力基本静态:calibrate() 一次,之后低频刷新即可。
    """

    def __init__(self, weights=MOGE_WEIGHTS, device=None):
        from moge.model.v2 import MoGeModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MoGeModel.from_pretrained(weights).eval().to(self.device)
        self.gravity = None  # (3,) 相机坐标系单位向量,指向"上"
        self.field = None  # (2, H, W) 仅用于画箭头的 2D 投影场
        self.points = None  # 最近一次点云 (H, W, 3)
        self.valid = None  # (H, W) bool
        self.inlier_ratio = None
        self.focal = None  # 从点图反推的焦距(像素),供 3D→2D 投影
        self.cx = self.cy = None

    @torch.inference_mode()
    def infer_points(self, frame_bgr):
        """返回 (points (H,W,3), valid (H,W) bool),相机坐标系(x右 y下 z前)。"""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        out = self.model.infer(t.to(self.device))
        pts = out["points"].float().cpu().numpy()
        valid = out["mask"].bool().cpu().numpy()
        self.points, self.valid = pts, valid
        return pts, valid

    @staticmethod
    def _plane_from_3(p0, p1, p2):
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            return None
        return n / norm

    def fit_ground_gravity(self, points, valid, bottom_frac=0.45, iters=300):
        """对画面下部区域 RANSAC 拟合地面平面,法向即重力(指向相机一侧为"上")。"""
        h, w = valid.shape
        ys, xs = np.nonzero(valid)
        keep = ys > int(h * (1.0 - bottom_frac))
        ys, xs = ys[keep], xs[keep]
        pts = points[ys, xs]
        if len(pts) < 500:
            raise RuntimeError("画面下部有效点太少,无法拟合地面")
        med_depth = float(np.median(pts[:, 2]))
        tol = max(0.02 * med_depth, 1e-4)

        rng = np.random.default_rng(0)
        best_inliers = None
        for _ in range(iters):
            i, j, k = rng.choice(len(pts), 3, replace=False)
            n = self._plane_from_3(pts[i], pts[j], pts[k])
            if n is None:
                continue
            inl = np.abs((pts - pts[i]) @ n) < tol
            if best_inliers is None or inl.sum() > best_inliers.sum():
                best_inliers = inl
        self.inlier_ratio = float(best_inliers.mean())
        if self.inlier_ratio < 0.35:
            raise RuntimeError(
                f"地面拟合内点率过低({self.inlier_ratio:.0%}),画面里可能没有可见地面"
            )

        # 内点上最小二乘精化:平面过质心,法向 = 协方差最小奇异向量
        c = pts[best_inliers].mean(axis=0)
        X = pts[best_inliers] - c
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        n = vt[-1]
        # 法向指向相机一侧(相机在原点):n·P < 0 ⇒ n 由地面指向相机 = "上"
        if float(n @ c) > 0:
            n = -n
        self.gravity = n.astype(np.float64)
        # 存下拟合结果供可视化:内点像素掩码 + 平面中心(内点质心)
        self.plane_center = c.astype(np.float64)
        self.ground_pixel_mask = np.zeros_like(valid)
        self.ground_pixel_mask[ys[best_inliers], xs[best_inliers]] = True
        return self.gravity

    def ground_inlier_mask(self, points, valid, tol_frac=0.02):
        """按已拟合的平面给任意一帧分类地面内点(无需重新 RANSAC)。"""
        if self.plane_center is None:
            raise RuntimeError("请先调用 fit_ground_gravity()")
        ys, xs = np.nonzero(valid)
        P = points[ys, xs]
        tol = tol_frac * float(np.median(np.abs(P[:, 2]))) + 1e-6
        inl = np.abs((P - self.plane_center) @ self.gravity) < tol
        m = np.zeros_like(valid)
        m[ys[inl], xs[inl]] = True
        return m

    def _estimate_focal(self, points, valid):
        """从点图反推焦距(像素):x = f·X/Z + cx,取双方向中位数。"""
        h, w = valid.shape
        ys, xs = np.nonzero(valid)
        P = points[ys, xs]
        z = P[:, 2]
        ok = np.abs(z) > 1e-6
        samples = []
        cx, cy = w / 2.0, h / 2.0
        dx = xs[ok] - cx
        m = np.abs(P[ok, 0]) > 0.05 * np.abs(z[ok])
        samples.append((dx[m] * z[ok][m] / P[ok, 0][m]))
        dy = ys[ok] - cy
        m = np.abs(P[ok, 1]) > 0.05 * np.abs(z[ok])
        samples.append((dy[m] * z[ok][m] / P[ok, 1][m]))
        f = float(np.median(np.concatenate(samples)))
        return max(f, 1.0)

    def build_2d_field(self, points, valid, gravity):
        """把 3D 重力投影成逐像素 2D 方向场(仅供画箭头)。"""
        h, w = valid.shape
        f = self._estimate_focal(points, valid)
        self.focal, self.cx, self.cy = f, w / 2.0, h / 2.0
        cx, cy = self.cx, self.cy
        ys, xs = np.nonzero(valid)
        P = points[ys, xs]
        Q = P + gravity  # 沿重力上移一点再投影,差向量即该处"上"的 2D 方向

        def proj(V):
            zz = np.clip(V[:, 2], 1e-6, None)
            return np.stack((f * V[:, 0] / zz + cx, f * V[:, 1] / zz + cy), axis=1)

        d = proj(Q) - proj(P)
        norm = np.clip(np.linalg.norm(d, axis=1, keepdims=True), 1e-9, None)
        field = np.zeros((2, h, w), np.float32)
        field[0, ys, xs] = (d / norm)[:, 0]
        field[1, ys, xs] = (d / norm)[:, 1]
        self.field = field

    def calibrate(self, frames, bottom_frac=0.45):
        """取一帧(或多帧取中值重力)完成:点云 → 地面拟合 → 缓存重力与 2D 场。"""
        gravities = []
        for f in frames:
            pts, valid = self.infer_points(f)
            gravities.append(self.fit_ground_gravity(pts, valid, bottom_frac))
        g = np.median(np.stack(gravities), axis=0)
        self.gravity = g / np.linalg.norm(g)
        pts, valid = self.infer_points(frames[-1])
        self.build_2d_field(pts, valid, self.gravity)
        return self

    def refresh(self, frame_bgr):
        """对新一帧刷新点云与 2D 场(重力沿用已拟合值)。分辨率变化时必须走这里。"""
        pts, valid = self.infer_points(frame_bgr)
        self.build_2d_field(pts, valid, self.gravity)
        return self

    @property
    def gravity3d(self):
        if self.gravity is None:
            raise RuntimeError("请先调用 calibrate()")
        return self.gravity

    def up_at(self, cx, cy):
        """取缓存的 2D 重力场在 (cx, cy) 处的向量,仅用于画箭头。"""
        if self.field is None:
            raise RuntimeError("请先调用 calibrate()")
        v = self.field[:, int(cy), int(cx)]
        n = float(np.linalg.norm(v))
        if n <= 1e-6:
            return np.array([0.0, -1.0], dtype=np.float32)
        return v / n

    def points_in_mask(self, mask, margin=None):
        """取 mask 内的有效 3D 点 (N,3)。margin:可选深度带过滤比例(无 SAM3 mask 时兜底用)。"""
        if self.points is None:
            raise RuntimeError("请先调用 infer_points()")
        pts = self.points[mask & self.valid] if self.valid is not None else self.points[mask]
        if margin is not None and len(pts) > 50:
            z = pts[:, 2]
            zmed = np.median(z)
            pts = pts[np.abs(z - zmed) < margin * max(zmed, 1e-6)]
        return pts

    def project_to_image(self, P):
        """3D 点投影回像素坐标。P: (3,) 或 (N,3) → (2,) 或 (N,2)。"""
        P = np.asarray(P, dtype=np.float64)
        Z = np.clip(P[..., 2], 1e-6, None)
        out = np.stack(
            (self.focal * P[..., 0] / Z + self.cx, self.focal * P[..., 1] / Z + self.cy),
            axis=-1,
        )
        return out

    def draw_axis(self, frame, pts, color=(0, 165, 255), thickness=3):
        """把瓶身点云的 PCA 轴向投影画到图上(两端各延伸 15% 便于目视)。橙色实线。"""
        pts = np.asarray(pts, dtype=np.float64)
        if len(pts) < 10 or not self.focal:
            return
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        a = vt[0]
        L = float(np.percentile(np.abs((pts - c) @ a), 98)) * 1.15
        q1 = self.project_to_image(c - a * L)
        q2 = self.project_to_image(c + a * L)
        cv2.line(frame, (int(q1[0]), int(q1[1])), (int(q2[0]), int(q2[1])), color, thickness)
        for q in (q1, q2):
            cv2.circle(frame, (int(q[0]), int(q[1])), thickness + 1, color, -1)

"""单张图测试:YOLO 检测 + SAM3 分割 + MoGe 3D 重力,判断氧气瓶是否倒伏。

用法:
  uv run python scripts/run_image.py --image path/to/img.jpg                 # 2D 保守管线(ParamNet)
  uv run python scripts/run_image.py --image img.jpg --moge --sam3           # 3D 主管线(推荐)
  uv run python scripts/run_image.py --image a.jpg b.jpg --save r.jpg --viz-field
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from ultralytics import YOLO

from paramnet_yolo import GravityEstimator, MoGeGravity, TippingJudge


def imread_unicode(path):
    """cv2.imread 在 Windows 中文路径下会返回 None,用 imdecode 规避。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def draw_up_arrow(img, center, up, length=60, color=(0, 255, 0), thickness=2):
    p1 = (int(center[0]), int(center[1]))
    p2 = (int(center[0] + up[0] * length), int(center[1] + up[1] * length))
    cv2.arrowedLine(img, p1, p2, color, thickness, tipLength=0.2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, nargs="+", help="图片路径(可多张)")
    ap.add_argument("--yolo", default="models/yolo.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--angle-thresh", type=float, default=45.0, help="轴向-重力夹角阈值(度)")
    ap.add_argument("--aspect-thresh", type=float, default=1.2, help="bbox 模式高宽比阈值")
    ap.add_argument("--moge", action="store_true", help="启用 MoGe 3D 管线(免疫径向盲区)")
    ap.add_argument("--sam3", action="store_true", help="启用 SAM3 分割过滤点云背景(配合 --moge)")
    ap.add_argument("--viz-field", action="store_true", help="叠加全图重力场箭头网格")
    ap.add_argument("--save", default=None, help="结果保存路径(多图时自动加序号)")
    args = ap.parse_args()

    model = YOLO(args.yolo)
    seg = None
    est = None

    def ensure_estimators(frame):
        """首个可用图片处初始化模型(避免坏图先触发加载)。"""
        nonlocal seg, est
        if est is not None:
            return
        t0 = time.time()
        if args.moge:
            est = MoGeGravity().calibrate([frame])
            print(f"MoGe 加载+地面拟合完成({time.time() - t0:.0f}s), 内点率 {est.inlier_ratio:.0%}")
        else:
            est = GravityEstimator().calibrate([frame])
            print(f"ParamNet 加载完成({time.time() - t0:.0f}s)")
        if args.sam3:
            t0 = time.time()
            from paramnet_yolo import Sam3Segmenter

            seg = Sam3Segmenter()
            print(f"SAM3 加载完成({time.time() - t0:.0f}s)")

    judge = TippingJudge(None, args.angle_thresh, args.aspect_thresh)

    for idx, img_path in enumerate(args.image):
        frame = imread_unicode(img_path)
        if frame is None:
            print(f"[跳过] 读不到图片: {img_path}")
            continue
        ensure_estimators(frame)
        judge.gravity = est

        if args.moge:
            est.refresh(frame)  # 刷新点云与 2D 场(重力沿用已拟合值)

        if args.viz_field:
            h, w = frame.shape[:2]
            for y in range(h // 2, h, 90):
                for x in range(45, w, 90):
                    draw_up_arrow(frame, (x, y), est.up_at(x, y), length=28, color=(255, 200, 0), thickness=1)

        result = model.predict(frame, conf=args.conf, retina_masks=True, verbose=False)[0]
        yolo_masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        obbs = result.obb

        if yolo_masks is not None:
            n_det = len(yolo_masks)
        elif obbs is not None:
            n_det = len(obbs)
        else:
            n_det = len(result.boxes)

        for i in range(n_det):
            if yolo_masks is not None:
                mask = (yolo_masks[i] > 0.5).astype(np.uint8)
                box = tuple(map(int, result.boxes.xyxy[i].tolist()))
            elif obbs is not None:
                pts_q = obbs.xyxyxyxy[i].cpu().numpy()
                box = (
                    int(pts_q[:, 0].min()), int(pts_q[:, 1].min()),
                    int(pts_q[:, 0].max()), int(pts_q[:, 1].max()),
                )
                mask = None
            else:
                box = tuple(map(int, result.boxes.xyxy[i].tolist()))
                mask = None

            if seg is not None:
                m = seg.set_image(frame).mask_from_box(*box)
                if m is not None:
                    mask = m.astype(np.uint8)
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, cnts, -1, (255, 0, 255), 2)  # 紫色轮廓=SAM3分割边界

            conf = float(result.boxes.conf[i]) if obbs is None else float(obbs.conf[i])
            cls_name = result.names[int(result.boxes.cls[i] if obbs is None else obbs.cls[i])]
            color = (0, 0, 255)

            if args.moge and mask is not None and mask.sum() >= 10:
                # 主管线:mask 内 3D 点 PCA vs 地面法向
                bottle_pts = est.points_in_mask(mask.astype(bool), margin=0.5)
                tipped, angle, elong = judge.judge_by_3d(bottle_pts)
                if tipped is None:
                    continue
                est.draw_axis(frame, bottle_pts)  # 橙色线=瓶子3D轴向
                detail = f"3D axis {angle:.0f}deg x{elong:.1f}"
            elif args.moge and mask is None:
                # 无 SAM3 的兜底:框内深度带过滤
                x1, y1, x2, y2 = box
                rough = np.zeros(est.valid.shape, bool)
                rough[y1:y2, x1:x2] = True
                bottle_pts = est.points_in_mask(rough, margin=0.15)
                tipped, angle, elong = judge.judge_by_3d(bottle_pts)
                if tipped is None:
                    continue
                detail = f"3D band {angle:.0f}deg"
            elif mask is not None:
                tipped, angle, center2d = judge.judge_by_mask(mask)
                if tipped is None:
                    continue
                detail = f"axis-up {angle:.0f}deg"
            else:
                tipped, aspect = judge.judge_by_bbox(box)
                detail = f"h/w {aspect:.2f}"

            x1, y1, x2, y2 = box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            draw_up_arrow(frame, center, est.up_at(*center))
            label = ("TIPPED " if tipped else "OK ") + detail + f" {conf:.2f}"
            if obbs is not None and yolo_masks is None:
                pts_q = obbs.xyxyxyxy[i].cpu().numpy().astype(np.int32)
                cv2.polylines(frame, [pts_q], True, (0, 200, 0) if not tipped else color, 2)
                top = pts_q[np.argmin(pts_q[:, 1])]
                label_xy = (int(top[0]), int(top[1]) - 6)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0) if not tipped else color, 2)
                label_xy = (x1, y1 - 6)
            cv2.putText(frame, label, label_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            print(f"  检测 {cls_name} conf={conf:.2f} -> {label}")

        if n_det == 0:
            print("  (未检出氧气瓶)")

        if args.save:
            out = args.save if len(args.image) == 1 else args.save.replace(
                Path(args.save).suffix, f"_{idx + 1}{Path(args.save).suffix}"
            )
            imwrite_unicode(out, frame)
            print(f"  已保存: {out}")
        else:
            cv2.imshow("result (任意键下一张)", frame)
            cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

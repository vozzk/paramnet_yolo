"""视频倒伏检测主循环:固定机位校准一次重力,每帧 YOLO(+SAM3)判断倒伏。

用法:
  uv run python scripts/run_video.py --source 0 --yolo models/yolo.pt                # 2D(ParamNet)
  uv run python scripts/run_video.py --source video.mp4 --moge --sam3                # 3D 主管线(推荐)
  uv run python scripts/run_video.py --source video.mp4 --moge --sam3 --save out.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paramnet_yolo import GravityEstimator, MoGeGravity, TippingJudge


def draw_up_arrow(img, center, up, length=60):
    p1 = (int(center[0]), int(center[1]))
    p2 = (int(center[0] + up[0] * length), int(center[1] + up[1] * length))
    cv2.arrowedLine(img, p1, p2, (0, 255, 0), 2, tipLength=0.2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="视频路径或摄像头序号(如 0)")
    ap.add_argument("--yolo", default="models/yolo.pt", help="YOLO 权重(自训氧气瓶模型)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--calib-frames", type=int, default=10, help="重力校准用帧数")
    ap.add_argument("--angle-thresh", type=float, default=45.0, help="轴向-重力夹角阈值(度)")
    ap.add_argument("--aspect-thresh", type=float, default=1.2, help="bbox 模式高宽比阈值")
    ap.add_argument("--moge", action="store_true", help="启用 MoGe 3D 管线(免疫径向盲区)")
    ap.add_argument("--sam3", action="store_true", help="启用 SAM3 分割过滤点云背景(配合 --moge)")
    ap.add_argument("--moge-every", type=int, default=10, help="3D 点云刷新间隔(帧);重力始终复用校准值")
    ap.add_argument("--save", default=None, help="可选,保存标注视频路径")
    args = ap.parse_args()

    cap = cv2.VideoCapture(
        int(args.source) if args.source.isdigit() else args.source
    )
    if not cap.isOpened():
        sys.exit(f"无法打开视频源: {args.source}")

    frames = []
    for _ in range(args.calib_frames):
        ret, frame = cap.read()
        if not ret:
            sys.exit("视频帧不足,无法完成重力校准")
        frames.append(frame)

    print("校准重力中(首次加载约需十几秒到一分钟)...")
    t0 = time.time()
    if args.moge:
        est = MoGeGravity().calibrate(frames)
        print(f"MoGe 完成({time.time() - t0:.0f}s),地面内点率 {est.inlier_ratio:.0%}")
    else:
        est = GravityEstimator().calibrate(frames)
        print(f"ParamNet 完成({time.time() - t0:.0f}s)")

    seg = None
    if args.sam3:
        from paramnet_yolo import Sam3Segmenter

        seg = Sam3Segmenter()
        print("SAM3 就绪")

    judge = TippingJudge(est, args.angle_thresh, args.aspect_thresh)
    model = YOLO(args.yolo)

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 3D 管线:按间隔刷新点云(重力恒用校准值);静止监控场景下隔帧点云对判断影响很小
        if args.moge and (frame_idx - 1) % max(args.moge_every, 1) == 0:
            est.refresh(frame)

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

            if args.moge and mask is not None and mask.sum() >= 10:
                bottle_pts = est.points_in_mask(mask.astype(bool), margin=0.5)
                tipped, angle, elong = judge.judge_by_3d(bottle_pts)
                if tipped is None:
                    continue
                est.draw_axis(frame, bottle_pts)  # 橙色线=瓶子3D轴向
                detail = f"3D {angle:.0f}deg"
            elif args.moge:
                x1, y1, x2, y2 = box
                rough = np.zeros(est.valid.shape, bool)
                rough[y1:y2, x1:x2] = True
                bottle_pts = est.points_in_mask(rough, margin=0.15)
                tipped, angle, elong = judge.judge_by_3d(bottle_pts)
                if tipped is None:
                    continue
                detail = f"3D band {angle:.0f}deg"
            elif mask is not None:
                tipped, angle, _ = judge.judge_by_mask(mask)
                if tipped is None:
                    continue
                detail = f"axis-up {angle:.0f}deg"
            else:
                tipped, aspect = judge.judge_by_bbox(box)
                detail = f"h/w {aspect:.2f}"

            x1, y1, x2, y2 = box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            draw_up_arrow(frame, center, est.up_at(*center))
            label = ("TIPPED " if tipped else "OK ") + detail
            color = (0, 0, 255) if tipped else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if writer is not None:
            writer.write(frame)
        cv2.imshow("oxygen tipping", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

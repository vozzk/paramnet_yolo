# paramnet_yolo — 氧气瓶倒伏检测

YOLO 检测氧气瓶 + SAM3 分割 + MoGe 单目几何,从 3D 点云判断瓶子是否倒伏;ParamNet(2D 重力场)作为无地面场景的 fallback 保留。

## 结构

```
paramnet_yolo/
├── paramnet_yolo/
│   ├── gravity.py        # MoGeGravity(3D主管线): 地面平面拟合→重力;GravityEstimator(ParamNet fallback)
│   ├── sam3_seg.py       # Sam3Segmenter: YOLO框→SAM3贴身mask
│   └── tipping.py        # TippingJudge: judge_by_3d(推荐)/seg轴向/OBB轴向/bbox高宽比
├── scripts/
│   ├── smoke_test.py     # ParamNet 离线加载自检
│   ├── run_image.py      # 单图: --moge --sam3 走3D主管线
│   └── run_video.py      # 视频: 同上, --moge-every 控制点云刷新间隔
└── models/               # 自训 YOLO 权重(yolo.pt, detect任务, 类别 oxygen_tank)
```

## 快速开始

```bash
uv sync                                            # 基础环境(torch 用本地 wheel)
uv pip install --no-deps -e E:\code\MoGe -e E:\code\utils3d-moge -e E:\code\FlexGEMM
                                                   # ⚠️ uv sync 会移除这三个包,需重跑此行
uv run python scripts/run_image.py --image "D:/PycharmProjects/yolo_26/ultralytics/test_image/9.jpg" --moge --sam3 --save result.jpg
uv run python scripts/export_pointcloud.py --image "D:/PycharmProjects/yolo_26/ultralytics/test_image/9.jpg"
```

## 三级判据(按可用性自动选择)

| 条件 | 判据 | 抗大俯拍角 | 抗径向躺倒盲区 |
|---|---|---|---|
| --moge(+--sam3 最佳) | **3D**: 瓶身点云 PCA 长轴 vs 地面法向夹角 | ✅ | ✅ |
| 有 mask(无 MoGe) | 2D: mask 长轴 vs 重力场夹角 | ✅ | ❌ |
| 仅检测框 | bbox 高宽比(启发式) | ❌ | ❌ |

**径向盲区**: 瓶子顺视线方向躺倒时,2D 投影轴向与重力夹角仅 4~15°(投影像素上"看起来像站着"),
2D 判据必然漏判;3D 点云直接恢复轴向与地面的真实夹角(躺倒≈90°),免疫此问题。

## 3D 管线原理

```
YOLO 检测框 → SAM3 box-prompt 分割(滤背景) → MoGe 点云取瓶身 3D 点 → PCA → 瓶子 3D 轴向
画面下部点云 → RANSAC 地面平面拟合 → 法向 = 重力方向(无需 ParamNet)
轴向·重力 夹角 > 45° → TIPPED
```

## 权重与代码位置(全部离线)

- MoGe-2-vitl: `E:\modelscope\hub\models\moge-2-vitl\model.pt`(1.3G;MoGe-3 需 MSVC+triton,Windows 未装,弃用)
- SAM3: `E:\modelscope\hub\models\sam3\sam3.pt`(3.3G),代码 `E:\code\sam3`(已打 Windows 兼容补丁:pkg_resources→importlib、视频跟踪 triton 链懒加载、edt cv2 回退)
- MoGe 代码: `E:\code\MoGe` + `E:\code\utils3d-moge` + `E:\code\FlexGEMM`(git 依赖经 codeload 拉取)
- ParamNet: `E:\torch\hub\checkpoints\`(TORCH_HOME 已设),源码 `E:\code\PerspectiveFields`
- YOLO: `models/yolo.pt`(detect, oxygen_tank, 来自 clinder_11m_0529/best.pt)

## 实测记录(2026-10-10, test_image/1-3.jpg)

- 地面拟合内点率 72%;站立瓶 3D 夹角 3°(伸长比 37),已知躺倒瓶 88°,躺倒瓶 69° → 判定全部正确
- 已知边界: 重度遮挡瓶(推车半掩)PCA 角度偏大(36°),可见点太少时伸长比会提示可信度
- SAM3+MoGe 加载约 20s;视频模式点云按 --moge-every 刷新,重力恒用校准值

## 重力场使用原则(2D fallback 仍适用)

- 固定机位校准一次全程复用;相机移动/转向/变焦/切分辨率后必须重新校准;
- 2D 重力场逐像素不同(通道 0=x、1=y,y 向下),判每个瓶子用它所在位置的向量。

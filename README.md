# 基于 YOLOv8 与 DeepSORT 的多目标跟踪系统

基于 **MOT17** 数据集的多目标跟踪系统，实现并对比 **SORT**、**DeepSORT**、**ByteTrack** 三种跟踪算法。

## 项目特点

- **三种算法对比**: SORT / DeepSORT / ByteTrack
- **MOT17 数据集**: 使用 MOTChallenge MOT17 标准数据集训练与评估
- **DeepSORT + YOLOv8**: 结合 Re-ID 外观特征与卡尔曼滤波的高精度跟踪
- **ByteTrack**: 利用高/低分检测两阶段关联的高效跟踪
- **GUI 界面**: 支持实时跟踪可视化
- **学术实验框架**: 算法横向对比、参数消融、检测器消融
- **数据图表可视化**: 分组柱状图、雷达图、消融折线图、热力图、误差饼图等
- **完整评估**: MOTA / MOTP / IDF1 / HOTA / MT / ML 等 MOTChallenge 标准指标

## 算法简介

### SORT

基线算法，使用卡尔曼滤波预测目标运动状态，通过 IoU 匹配实现帧间关联，采用匈牙利算法求解最优匹配。优点是速度快，缺点是在遮挡场景下容易产生 ID 切换。

### DeepSORT

在 SORT 基础上引入 Re-ID 外观特征描述子，将运动信息与外观信息融合进行数据关联，并采用级联匹配策略优先匹配最近可见的轨迹，显著降低了 ID 切换率。

### ByteTrack

创新性地利用低置信度检测框：第一阶段将高分检测与现有轨迹匹配，第二阶段将未匹配轨迹与低分检测进行二次关联，从而在拥挤、遮挡等困难场景中保持更稳定的跟踪。

## 实验设计

| 实验 | 内容 | 目的 |
|------|------|------|
| 实验 1 | SORT vs DeepSORT vs ByteTrack | 三种核心算法横向对比 |
| 实验 2 | DeepSORT 参数消融 | 关键参数对性能的影响 |
| 实验 3 | 检测器消融 | 检测质量对跟踪的影响 |

### 评估指标

| 指标 | 说明 |
|------|------|
| **MOTA** | 多目标跟踪准确度 |
| **MOTP** | 多目标跟踪精确度，衡量检测框与 GT 的重合度 |
| **IDF1** | ID F1 分数，衡量身份保持能力 |
| **HOTA** | 综合检测与关联质量的统一指标 |
| **MT** | Mostly Tracked，≥80% 生命周期被跟踪的目标比例 |
| **ML** | Mostly Lost，≤20% 生命周期被跟踪的目标比例 |
| **FP / FN** | 误漏检数量 |
| **IDSW** | ID 切换次数 |

## 环境配置

### 硬件

- **训练设备**: NVIDIA RTX 5090

### 安装

```bash
# 创建虚拟环境
conda create -n mot python=3.10 -y
conda activate mot

# 安装依赖
pip install -r requirements.txt
```

### 依赖

- Python ≥ 3.10
- PyTorch ≥ 2.0
- Ultralytics
- OpenCV
- NumPy / SciPy
- filterpy
- lap
- torchreid
- PyQt5 / PySide6

## 数据集准备

### MOT17

从 [MOTChallenge](https://motchallenge.net/data/MOT17/) 下载 MOT17 数据集，解压至 `data/` 目录：

```bash
data/
└── MOT17/
    ├── train/
    │   ├── MOT17-02-SDP/
    │   ├── MOT17-04-SDP/
    │   ├── MOT17-05-SDP/
    │   ├── MOT17-09-SDP/
    │   ├── MOT17-10-SDP/
    │   ├── MOT17-11-SDP/
    │   └── MOT17-13-SDP/
    └── test/
        ├── MOT17-01-SDP/
        ├── MOT17-03-SDP/
        ├── MOT17-06-SDP/
        ├── MOT17-07-SDP/
        ├── MOT17-08-SDP/
        ├── MOT17-12-SDP/
        └── MOT17-14-SDP/
```

> 每个序列包含 `seqinfo.ini` 元信息、`det/det.txt` 公共检测、`img1/` 视频帧，训练集还包含 `gt/gt.txt` 标注。

### MOT20

MOT20 与 MOT17 数据目录结构完全一致, 同为 MOTChallenge 格式, 项目可直接读取。由于 YOLOv8m 仅在 MOT17 上 fine-tune, 在 MOT20 上运行时属于跨域测试, 可用来观察算法在稠密人群下的鲁棒性。

从 [MOTChallenge](https://motchallenge.net/data/MOT20/) 下载并解压至 `data/MOT20/`:

```bash
data/
└── MOT20/
    ├── train/                  # MOT20-01 / 02 / 03 / 05
    └── test/                   # MOT20-04 / 06 / 07 / 08
```

## 模型权重准备

项目使用 **YOLOv8m** 作为检测器、**OSNet x0.25** 作为 Re-ID 外观特征提取器。检测器需要在 MOT17 上微调。

```bash
# 第一步: 将 MOT17 GT 转换为 Ultralytics YOLO 格式
python scripts/convert_mot17_to_yolo.py \
    --mot-root data/MOT17 \
    --out-root data/MOT17_yolo \
    --val-ratio 0.2           # 每条序列末尾 20% 帧作为验证集

# 第二步: 从 COCO 预训练权重开始微调 YOLOv8m
python scripts/train_yolov8_mot17.py \
    --data data/MOT17_yolo/dataset.yaml \
    --weights models/yolov8m.pt \
    --epochs 50 \
    --batch 16 \
    --imgsz 1280 \
    --device 0
```

训练完成后 `best.pt` 会自动复制到 [models/yolov8m_mot17.pt](models/)，配置已默认指向该权重。

**训练配方要点**:

| 项目 | 取值 | 说明 |
|------|------|------|
| 初始权重 | `models/yolov8m.pt` | 迁移学习起点，person 类已有良好先验 |
| 图像尺寸 | `1280` | MOT17 为 1920×1080，保留小目标行人信息 |
| Batch Size | `16` | RTX 5090 32GB 可安全承载；如有余量可调至 24 |
| Epochs | `50` | 上一轮训练在第 29 轮达到最佳, 50 轮已足够收敛 |
| 优化器 | SGD + cosine LR | `lr0=0.01`, `momentum=0.937`, `wd=5e-4` |
| Warmup | `3 epochs` | 稳定早期梯度 |
| 数据增强 | mosaic + mixup + copy-paste | 最后 10 epoch 关闭 mosaic 对齐原分布 |
| AMP | 开启 | 混合精度，利用 5090 FP16 / TF32 Tensor Core |
| 缓存 | `ram` | MOT17 体量小，RAM 缓存可大幅加速迭代 |

**预期指标 (MOT17 val, 单类 person)**:

| 指标 | 目标范围 | 说明 |
|------|----------|------|
| AP50 | ≥ 0.90 | 主流 YOLOX-m / YOLOv7 基线在 0.85~0.90 之间 |
| AP50-95 | ≥ 0.60 | 反映定位精度 |
| Recall | ≥ 0.88 | 高召回对后续跟踪关联至关重要 |

## 数据图表可视化

实验运行完毕后，系统自动在 `outputs/plots/` 目录生成以下图表：

| 图表类型 | 说明 | 来源实验 |
|----------|------|----------|
| 分组柱状图 | MOTA / IDF1 / IDSW 多指标并列对比 | 实验 1 |
| 雷达图 | 算法多维性能归一化蛛网图 | 实验 1 |
| FPS 横向条形图 | 跟踪速度对比 | 实验 1 / 3 |
| 逐序列柱状图 | 各序列下算法表现 | 实验 1 |
| 热力图 | Tracker × Sequence 指标矩阵 | 实验 1 |
| 误差饼图 | FP / FN / IDSW 占比分布 | 实验 1 |
| 消融折线图 | 参数值变化对 MOTA / IDF1 的影响 | 实验 2 |
| 消融分组柱状图 | Re-ID 开关等分类变量对比 | 实验 2 |
| 检测器对比柱状图 | 不同检测器下跟踪准确率 | 实验 3 |

## 使用方法

### 快速准备

```bash
# 1. 下载并放置 MOT17 数据集到 data/MOT17/ (见上文)
# 2. 下载官方 Re-ID 权重
python scripts/download_reid_weights.py --model osnet_x0_25_msmt17
# 3. (可选) 微调 YOLOv8m 检测器
python scripts/convert_mot17_to_yolo.py
python scripts/train_yolov8_mot17.py
```

### 运行跟踪

```bash
# 使用 main.py 统一入口 (MOT17)
python main.py track --config configs/deepsort.yaml --sequence MOT17-02-SDP
python main.py track --config configs/sort.yaml --sequence MOT17-09-SDP
python main.py track --config configs/bytetrack.yaml --sequence MOT17-09-SDP

# MOT20: 使用专用配置, 或用 --data-root 覆盖 MOT17 配置
python main.py track --config configs/deepsort_mot20.yaml --sequence MOT20-01
python main.py track --config configs/deepsort.yaml \
    --data-root data/MOT20 --sequence MOT20-01
```

### 运行实验

实验结束后会自动在 `outputs/exp*/plots/` 下生成可视化图表。

```bash
# 实验 1: 算法横向对比 (MOT17)
python main.py experiment --exp 1

# 实验 2: DeepSORT 参数消融
python main.py experiment --exp 2

# 实验 3: 检测器消融
python main.py experiment --exp 3

# 运行全部实验 (MOT17)
python main.py experiment --exp all

# 在 MOT20 上运行实验 (跨域测试)
python main.py experiment --exp 1 --data-root data/MOT20
python main.py experiment --exp all --data-root data/MOT20
```

### 对任意视频做实时跟踪

适用于非 MOTChallenge 格式的测试视频(如 `data/videos/*.mp4`)。使用 YOLOv8 检测器在线检测,配合指定的跟踪器输出带 ID 的可视化视频到 `outputs/videos/{视频名}/`。

```bash
# 单一跟踪器: 默认用 DeepSORT 处理 data/videos/ 下所有视频
python main.py video

# 指定单个视频 + 指定跟踪器
python main.py video --input data/videos/test_video_1.mp4 --tracker bytetrack

# 三算法横向对比: 输出 SORT/DeepSORT/ByteTrack 三个独立 mp4 + 一个 comparison.mp4 (并排)
python main.py video --input data/videos --compare

# 实时预览 (按 ESC 退出)
python main.py video --input data/videos/test_video_1.mp4 --tracker deepsort --show
```

**输出结构:**

```
outputs/videos/test_video_1/
├── DeepSORT.mp4       # 单算法 / --compare 模式下各算法分别输出
├── SORT.mp4           # (仅 --compare 时)
├── ByteTrack.mp4      # (仅 --compare 时)
├── comparison.mp4     # (仅 --compare 时) 三路并排对比
└── metrics.csv        # 每帧 track 数与瞬时 FPS, 便于后续分析
```

> 视频模式使用 YOLOv8 做在线检测(默认 `models/yolov8m_mot17.pt`);可用 `--detector yolov8s` 或 `--weights your.pt` 切换。

### 启动 GUI

```bash
python main.py gui
```

GUI 支持:
- **一键切换数据集**: 左上角 `MOT17` / `MOT20` 按钮,或用 `浏览 MOT 根目录...` 选任意路径
- **视频文件模式**: 点 `视频文件...` 选择 `data/videos/*.mp4` 等任意视频, 自动切换到 YOLOv8m 检测器进行在线跟踪
- 播放 / 暂停 / 单步 / 速度 / 拖动进度条
- 导出带跟踪框的视频 (`📁 导出视频`)

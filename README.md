# 实验 1 方法演示demo（默认 800 张）

这是 [Burgert 等人的论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/57a9b97477b67936298489e3c1417b0a-Paper-Conference.pdf)中实验 1 的可运行**方法演示**。它参考[作者开源仓库](https://github.com/tomburgert/feature-reliance)的图像处理和评分思路，直接使用三个公开的预训练权重，不训练或微调。

**注意：这不是论文实验1图 3 数值的严格复现。** 原论文作者在ImageNet中通过人工筛选选择每类 50 张、共 800 张图片且作者仓库未提供这份图片清单，其也未提供其自行训练的ResNet50-standard checkpoint故。本项目改用公开的 [ImageNet16 验证集](https://zenodo.org/records/8027520)，一共16类，按文件名每类取前 50 张；另把 [`timm/resnet50.tv_in1k`](https://huggingface.co/timm/resnet50.tv_in1k) 明确标为 **ResNet50-standard 代理模型**，另两个模型（依据原作者的描述）选用timm 预训练 checkpoint，分别是 [`timm/resnet50.a1_in1k`](https://huggingface.co/timm/resnet50.a1_in1k) 和 [`timm/convnextv2_tiny.fcmae_ft_in22k_in1k`](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k)。结果适合检查代码流程、比较趋势，不能按论文数值或排序验收。

## 电脑配置

- 2021 款 M1 MacBook Air、8 GB 内存可以运行。程序一次只载入一个模型，默认批量大小为 2；自动选择 MPS，若不可用则使用 CPU。首次运行的实际时间依网络、设备和 PyTorch 后端而异。
- macOS **14 或更新版本**的 Apple Silicon，Python **3.12**；本仓库在 macOS 15.2 的 M1 上验证。建议至少预留 **4 GB** 可用磁盘空间。虚拟环境、权重和处理后的图片占主要空间。系统需要联网下载 Python 包、模型和图片。
- 默认数据为每类 50 张、16 类共 **800 张**；每个模型评估 5 种条件，共 12,000 张次推理。

本机实测：macOS 15.2、M1、8 GB 内存、MPS，依赖与权重已缓存、首次获取 800 张图片时，全流程 **390 秒（6 分 30 秒）**；其中三个模型评估约 57、59、97 秒。进程峰值 RSS 约 **679 MiB**（不等于全机内存，未单独统计 MPS 驱动分配）；虚拟环境占用约 **970 MiB**。这些是这次运行的记录，不是其他机器的耗时保证。

## 一条命令运行

在本目录执行：

```bash
./run.sh
```

`run.sh` 自动建立 `.venv`、安装 `requirements-lock.txt` 中固定版本的 Python 包，然后下载资源、处理图片并评估。`requirements.txt` 列出直接依赖；锁文件同时固定间接依赖。首次需联网；再次运行复用已下载的权重和处理缓存。运行时终端会输出每个模型、每种条件的准确率。

可选命令：

```bash
./run.sh --smoke       # 每类 1 张，先检查全流程
./run.sh --device cpu  # 强制 CPU
./run.sh --batch-size 1
```

运行 `./run.sh --help` 可查看数据目录、权重目录、输出目录等参数。已有完整 Zenodo ZIP 时可传 `--data-zip /绝对路径/imagenet16.zip`；程序校验下载页公布的 MD5 后使用它。

## 数据与处理

默认按 [Zenodo ImageNet16 ZIP](https://zenodo.org/records/8027520) 的每个验证集 WNID 文件夹，**按文件名排序并取前 50 张**。这不是随机抽样；结果中仍记录随机种子 `1`，用于 Patch Shuffle 的固定打乱顺序。仓库追踪 `config/sample_manifest_800.csv`，列出 800 张图片的 ZIP 内路径、WNID 和大类，不包含原图。每次运行都会核对实际选图与这份清单；`--smoke` 使用各类清单的第 1 张。

Zenodo 原始 ZIP 约 885 MB。服务器支持 HTTP Range，程序只下载 ZIP 索引和被选图片的字节区间；选中 ZIP 条目逐个用 CRC 校验。磁盘上的稀疏 ZIP 文件显示逻辑大小约 885 MB，实际占用只相当于所下载的区间。`--data-zip` 模式会校验完整 ZIP 的 MD5。若远端不再支持 Range，请先手动下载完整 ZIP 并用 `--data-zip` 指定。

每张图片先缩放到 224×224，再生成五种条件；生成后的同一份图片缓存供三个模型使用：

| 输出名 | 操作 |
| --- | --- |
| `original` | 原图缩放 |
| `global_shape` | 3×3 Patch Shuffle |
| `local_shape` | 6×6 Patch Shuffle |
| `texture` | 双边滤波：`d=12, sigmaColor=170, sigmaSpace=75` |
| `color` | 灰度后复制为 RGB 三通道 |

细类别到 16 大类的对应关系来自 [Geirhos 等人的公开映射](https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/16-class-ImageNet/MSCOCO_to_ImageNet_category_mapping.txt)，随仓库存于 `config/imagenet16_category_map.txt`。按论文的方法，对**真实大类**所含的 ImageNet-1k 子类别的 softmax 概率求和；严格大于 `0.5` 判为正确。原始准确率为正确图片数除以图片总数；相对准确率为该条件的原始准确率除以同模型原图准确率。原图准确率为零时，相对准确率记为 `null`。

## 结果文件

默认保存在 `results/demo_800/`：

- `sample_manifest.csv`：所用 800 张图的路径、WNID 和大类，与仓库追踪的 `config/sample_manifest_800.csv` 一致；
- `metrics.csv`：3 个模型 × 5 个条件，共 15 行的正确数、原始准确率和相对准确率；
- `predictions.csv`：逐图的大类概率及判定，便于检查；
- `category_metrics.csv`：每个模型、条件和图片大类的正确数与准确率；
- `report.json`：方法参数、数据来源与校验方式、真实模型仓库修订号和 SHA-256、运行环境及耗时；
- `comparison.png`：原始准确率与相对准确率对比图。
- `category_comparison.png`：四种抑制条件的逐大类准确率图；点为各类准确率，竖线为该模型在所有图片上的准确率。图中只含三个 CNN，没有 human 数据。

已有 `predictions.csv` 和 `metrics.csv` 时，无需重新推理即可单独重画逐类图：

```bash
.venv/bin/python plot_categories.py results/demo_800
```

`--smoke` 输出到 `results/demo_16/`。这些结果和 `data/`、`.venv/` 都被 `.gitignore` 排除，不随 Git 仓库分发。

## 复现范围与来源

依赖版本在 `requirements-lock.txt` 固定。`config/model_weights.json` 固定三个 Hugging Face 仓库的提交修订号和 SHA-256；程序从该不可变修订号下载 `model.safetensors` 并核对哈希。`report.json` 同时记录本次使用的修订号与哈希。若远端移除这些提交，需选择仍可获取的权重、更新锁文件，并重新说明结果差异。

实验方法见[论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/57a9b97477b67936298489e3c1417b0a-Paper-Conference.pdf)及[官方补充材料](https://proceedings.neurips.cc/paper_files/paper/2025/file/57a9b97477b67936298489e3c1417b0a-Supplemental-Conference.zip)。本仓库只实现实验 1 的推理与统计，不包含原论文的人工筛选图片、训练过程或原始 checkpoint。

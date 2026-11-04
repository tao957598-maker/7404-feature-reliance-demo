# 实验 1 方法演示 demo（Apple Silicon macOS / Windows x64，默认 800 张）

**Experiment 1 Method Demo (Apple Silicon macOS / Windows x64, 800 images by default)**

这是 [Burgert 等人的论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/57a9b97477b67936298489e3c1417b0a-Paper-Conference.pdf)中实验 1 的可运行**方法演示**。它参考[作者开源仓库](https://github.com/tomburgert/feature-reliance)的图像处理和评分思路，直接使用三个公开的预训练权重，不训练或微调。

This runnable demo illustrates the method used in Experiment 1 of [Burgert et al.'s paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/57a9b97477b67936298489e3c1417b0a-Paper-Conference.pdf). It follows the image transformations and scoring approach in the [authors' open-source repository](https://github.com/tomburgert/feature-reliance) and evaluates three publicly available pretrained checkpoints. It does not train or fine-tune any models.

**注意：这不是论文实验 1 图 3 数值的严格复现。** 原论文作者从 ImageNet 人工筛选每类 50 张、共 800 张图片，但作者仓库未提供图片清单，也未提供其自行训练的 ResNet50-standard checkpoint。本项目改用公开的 [ImageNet16 验证集](https://zenodo.org/records/8027520)，共 16 类，按文件名每类取前 50 张；另把 [`timm/resnet50.tv_in1k`](https://huggingface.co/timm/resnet50.tv_in1k) 明确标为 **ResNet50-standard 代理模型**，另两个模型（依据原作者的描述）选用 timm 预训练 checkpoint，分别是 [`timm/resnet50.a1_in1k`](https://huggingface.co/timm/resnet50.a1_in1k) 和 [`timm/convnextv2_tiny.fcmae_ft_in22k_in1k`](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k)。结果适合检查代码流程、比较趋势，不能按论文数值或排序验收。

**This demo does not reproduce the exact numbers in Figure 3 of the paper.** The authors curated 50 ImageNet images per class, for a total of 800, but their repository does not provide the image list or their original ResNet50-standard checkpoint. This demo instead takes the first 50 images by filename from each of the 16 classes in the public [ImageNet16 validation set](https://zenodo.org/records/8027520). It uses three timm checkpoints:

- [`timm/resnet50.tv_in1k`](https://huggingface.co/timm/resnet50.tv_in1k), labeled **ResNet50-standard (proxy)**.
- [`timm/resnet50.a1_in1k`](https://huggingface.co/timm/resnet50.a1_in1k).
- [`timm/convnextv2_tiny.fcmae_ft_in22k_in1k`](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k).

The latter two checkpoints were selected based on the authors' descriptions. Use the results to check the workflow and compare trends on this dataset. The exact scores and model rankings may differ from the paper.

## 电脑配置

**System requirements**

- 使用 **64 位 Python 3.12**。当前固定的 PyTorch 版本支持 Apple Silicon Mac 和 x86-64 Windows 电脑；Intel Mac 与 Windows ARM 未纳入本项目的兼容范围。程序一次只载入一个模型，默认批量大小为 2。自动选择可用的 CUDA、Apple MPS 或 CPU，也可以用 `--device cpu` 强制使用 CPU。Windows 默认使用 1 个 PyTorch CPU 线程；macOS 最多使用 4 个。

  Use **64-bit Python 3.12**. The pinned PyTorch version supports Apple Silicon Macs and x86-64 Windows PCs. Intel Macs and Windows on ARM are outside the scope of this project. The program loads one model at a time and defaults to a batch size of 2. It automatically selects CUDA, Apple MPS, or CPU based on availability; use `--device cpu` to force CPU inference. PyTorch uses one CPU thread on Windows and up to four on macOS.

- 本仓库已在 macOS 15.2 的 M1 MacBook Air（8 GB 内存）运行。Windows 建议使用 Windows 10/11 的 64 位版本。macOS 建议至少预留 **4 GB** 可用磁盘空间；Windows 建议至少 **6 GB**，因为稀疏 ZIP 可能占用完整文件大小。首次运行需要联网下载 Python 包、模型权重和图片。Windows 上的性能和内存占用可能不同。

  The demo has been run on an M1 MacBook Air with 8 GB of RAM and macOS 15.2. On Windows, 64-bit Windows 10 or 11 is recommended. Allow at least **4 GB** of free disk space on macOS and **6 GB** on Windows, where the sparse ZIP may occupy its full size on disk. The first run needs internet access to download Python packages, model weights, and images. Performance and memory use may differ on Windows.

- 默认数据为每类 50 张、16 类共 **800 张**；每个模型评估 5 种条件，共 12,000 张次推理。

  A full run uses **800 images** (50 per class across 16 classes). Evaluating three models under five conditions gives 12,000 image evaluations in total.

本机实测：macOS 15.2、M1、8 GB 内存、MPS，依赖与权重已缓存、首次获取 800 张图片时，全流程 **390 秒（6 分 30 秒）**；其中三个模型评估约 57、59、97 秒。进程峰值 RSS 约 **679 MiB**（不等于全机内存，未单独统计 MPS 驱动分配）；虚拟环境占用约 **970 MiB**。这些是这次运行的记录，不是其他机器的耗时保证。

In one run on the M1 MacBook Air using MPS, with dependencies and weights already cached but the 800 images fetched for the first time, the full workflow took **390 seconds (6 min 30 s)**. The three model evaluations took about 57, 59, and 97 seconds. Peak process RSS was about **679 MiB**; this is not total system memory usage, and MPS driver allocations were not measured separately. The virtual environment used about **970 MiB** of disk space. These figures describe this run only.

## macOS（Apple Silicon）运行方法

**Run on macOS (Apple Silicon)**

先安装 [Python 3.12](https://www.python.org/downloads/) 和 Git，在“终端”中运行：

Install [Python 3.12](https://www.python.org/downloads/) and Git, then run these commands in Terminal:

```bash
git clone https://github.com/tao957598-maker/7404-feature-reliance-demo.git
cd 7404-feature-reliance-demo
./run.sh --smoke
./run.sh
```

`--smoke` 每类只运行 1 张，先检查环境和完整流程；不带参数时运行 800 张。若已经在仓库目录中，可以直接从 `./run.sh --smoke` 开始。也可以使用 `python3.12 run.py` 替代 `./run.sh`。

The `--smoke` run uses one image per class to check the environment and the full workflow. Running without options evaluates 800 images. If you are already in the repository directory, start with `./run.sh --smoke`. You can also use `python3.12 run.py` in place of `./run.sh`.

## Windows（x86-64）运行方法

**Run on Windows (x86-64)**

安装 [64 位 Python 3.12](https://www.python.org/downloads/windows/)（安装时勾选 **Add python.exe to PATH**）和 [Git for Windows](https://git-scm.com/download/win)。打开 **PowerShell**，运行：

Install [64-bit Python 3.12](https://www.python.org/downloads/windows/) (select **Add python.exe to PATH** during setup) and [Git for Windows](https://git-scm.com/download/win). In **PowerShell**, run:

```powershell
git clone https://github.com/tao957598-maker/7404-feature-reliance-demo.git
cd 7404-feature-reliance-demo
py -3.12 run.py --smoke
py -3.12 run.py
```

如果电脑没有 `py` 命令，但 `python --version` 显示 3.12，可将上述两处 `py -3.12` 改成 `python`。无需激活虚拟环境，也无需运行 PowerShell 脚本。程序会在 Windows 中使用 `.venv\Scripts\python.exe`；macOS 使用 `.venv/bin/python`。

If the `py` command is unavailable but `python --version` shows Python 3.12, replace both occurrences of `py -3.12` with `python`. You do not need to activate the virtual environment or run a PowerShell script. The program uses `.venv\Scripts\python.exe` on Windows and `.venv/bin/python` on macOS.

**Windows CPU 注意：**早期版本在 Windows 上使用 4 个 PyTorch CPU 线程时，两种 ResNet 曾产生异常大的 logit，准确率接近零；此前生成的 Windows 图表不能用于汇报。当前版本将 Windows 的 CPU 线程数设为 1，并在 logit 或原图准确率异常时停止运行。`--batch-size 1` 调整的是每批图片数，不会改变 CPU 线程数。单线程可能增加运行时间。若已克隆旧版，请先在仓库目录执行 `git pull`，再按上面的命令重跑；已经下载的图片和权重缓存可以继续使用。

**Windows CPU note:** In an earlier version, using four PyTorch CPU threads on Windows caused both ResNet models to produce unusually large logits and near-zero accuracy. Do not use charts from those runs in a report or presentation. The current version uses one CPU thread on Windows and stops if it detects abnormal logits or accuracy on the original images. The option `--batch-size 1` controls the number of images per batch, not the number of CPU threads. Using one CPU thread may take longer. If you cloned an older version, run `git pull` from the repository directory, then repeat the commands above. You can reuse previously downloaded images and weights.

两个系统共用的 `run.py` 会自动建立 `.venv`、安装 `requirements-lock.txt` 中固定版本的 Python 包，然后下载资源、处理图片并评估。`requirements.txt` 列出直接依赖；锁文件同时固定间接依赖。再次运行复用已下载的权重和处理缓存。运行时终端会输出每个模型、每种条件的准确率。

On both platforms, `run.py` creates `.venv`, installs the pinned packages from `requirements-lock.txt`, downloads resources, processes the images, and runs the evaluation. The file `requirements.txt` lists direct dependencies; the lock file pins indirect dependencies as well. Later runs reuse downloaded weights and cached images. The program prints accuracy for each model and condition as it runs.

运行结束后可额外检查结果文件与准确率是否合理。macOS 使用 `python3.12 validate_results.py results/demo_800`；Windows PowerShell 使用 `py -3.12 validate_results.py results\demo_800`。这只是针对本仓库固定图片及权重的故障检查，不代表与论文数值相符。

After the run, you can check whether the output files and accuracy values look reasonable. On macOS, use `python3.12 validate_results.py results/demo_800`; on Windows PowerShell, use `py -3.12 validate_results.py results\demo_800`. This check catches problems with this repository's fixed inputs and weights; it does not check for agreement with the paper.

可选命令：

Optional commands:

```bash
./run.sh --device cpu  # 强制 CPU
./run.sh --batch-size 1
```

在 Windows PowerShell 中，将上述 `./run.sh` 改为 `py -3.12 run.py`。运行 `./run.sh --help` 或 `py -3.12 run.py --help` 可查看数据目录、权重目录、输出目录等参数。已有完整 Zenodo ZIP 时可传 `--data-zip` 和 ZIP 文件的绝对路径；程序校验下载页公布的 MD5 后使用它。

In Windows PowerShell, replace `./run.sh` with `py -3.12 run.py`. Run `./run.sh --help` or `py -3.12 run.py --help` to see options for data, weights, and output directories. If you already have the complete Zenodo ZIP, pass its absolute path with `--data-zip`. The program checks the archive against the MD5 checksum published on the download page before using it.

## 数据与处理

**Data and image transformations**

默认按 [Zenodo ImageNet16 ZIP](https://zenodo.org/records/8027520) 的每个验证集 WNID 文件夹，**按文件名排序并取前 50 张**。这不是随机抽样；结果中仍记录随机种子 `1`，用于 Patch Shuffle 的固定打乱顺序。仓库追踪 `config/sample_manifest_800.csv`，列出 800 张图片的 ZIP 内路径、WNID 和大类，不包含原图。每次运行都会核对实际选图与这份清单；`--smoke` 使用各类清单的第 1 张。

For each validation WNID folder in the [Zenodo ImageNet16 ZIP](https://zenodo.org/records/8027520), the default run **sorts images by filename and takes the first 50**. This is not random sampling. The recorded seed, `1`, is used to make Patch Shuffle repeatable. The tracked file `config/sample_manifest_800.csv` lists the archive path, WNID, and broad category for each of the 800 images. It does not contain the images themselves. The program checks each run's selection against this manifest. The `--smoke` run uses the first listed image in each class.

Zenodo 原始 ZIP 约 885 MB。服务器支持 HTTP Range，程序只下载 ZIP 索引和被选图片的字节区间；选中 ZIP 条目逐个用 CRC 校验。磁盘上的 ZIP 文件显示逻辑大小约 885 MB；实际占用空间取决于文件系统对稀疏文件的支持。`--data-zip` 模式会校验完整 ZIP 的 MD5。若远端不再支持 Range，请先手动下载完整 ZIP 并用 `--data-zip` 指定。

The original Zenodo ZIP is about 885 MB. The server supports HTTP Range requests, so the program downloads only the ZIP index and the byte ranges for the selected images. Each selected ZIP entry is checked against its CRC. The local ZIP has a logical size of about 885 MB, but its actual disk usage depends on whether the file system supports sparse files. With `--data-zip`, the program checks the MD5 of the entire ZIP. If the server stops supporting range requests, download the full ZIP manually and pass it with `--data-zip`.

每张图片先缩放到 224×224，再生成五种条件；生成后的同一份图片缓存供三个模型使用：

Each image is resized to 224×224, then prepared in five conditions. All three models use the same cached set of processed images.

| 输出名<br>Output name | 操作<br>Operation |
| --- | --- |
| `original` | 原图缩放<br>Resize the original image |
| `global_shape` | 3×3 Patch Shuffle |
| `local_shape` | 6×6 Patch Shuffle |
| `texture` | 双边滤波：`d=12, sigmaColor=170, sigmaSpace=75`<br>Bilateral filtering: `d=12, sigmaColor=170, sigmaSpace=75` |
| `color` | 灰度后复制为 RGB 三通道<br>Convert to grayscale and copy it into all three RGB channels |

细类别到 16 大类的对应关系来自 [Geirhos 等人的公开映射](https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/16-class-ImageNet/MSCOCO_to_ImageNet_category_mapping.txt)，随仓库存于 `config/imagenet16_category_map.txt`。按论文的方法，对**真实大类**所含的 ImageNet-1k 子类别的 softmax 概率求和；严格大于 `0.5` 判为正确。原始准确率为正确图片数除以图片总数；相对准确率为该条件的原始准确率除以同模型原图准确率。原图准确率为零时，相对准确率记为 `null`。

The mapping from ImageNet-1k classes to the 16 broad categories comes from [Geirhos et al.'s public mapping](https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/16-class-ImageNet/MSCOCO_to_ImageNet_category_mapping.txt) and is stored in `config/imagenet16_category_map.txt`. Following the paper, the program adds the softmax probabilities of the ImageNet-1k subclasses that belong to the **ground-truth broad category**. A prediction is counted as correct only if that sum is strictly greater than `0.5`.

Accuracy is the number of correct images divided by the total number of images. Relative accuracy is the accuracy for a given condition divided by the same model's accuracy on the original images. If the original-image accuracy is zero, the relative accuracy is recorded as `null`.

## 结果文件

**Output files**

默认保存在 `results/demo_800/`：

A full run writes its results to `results/demo_800/` by default:

- `sample_manifest.csv`：所用 800 张图的路径、WNID 和大类，与仓库追踪的 `config/sample_manifest_800.csv` 一致；

  `sample_manifest.csv`: archive paths, WNIDs, and broad categories for the 800 images, matching the tracked `config/sample_manifest_800.csv`.

- `metrics.csv`：3 个模型 × 5 个条件，共 15 行的正确数、原始准确率和相对准确率；

  `metrics.csv`: 15 rows (three models × five conditions), with correct counts, accuracy, and relative accuracy.

- `predictions.csv`：逐图的大类概率及判定，便于检查；

  `predictions.csv`: per-image probability for the ground-truth broad category and the resulting correct/incorrect decision.

- `category_metrics.csv`：每个模型、条件和图片大类的正确数与准确率；

  `category_metrics.csv`: correct counts and accuracy for each model, condition, and broad category.

- `report.json`：方法参数、数据来源与校验方式、真实模型仓库修订号和 SHA-256、运行环境、PyTorch CPU 线程数及耗时；

  `report.json`: transformation settings, data sources and verification methods, model repository revisions and SHA-256 checksums, run environment, PyTorch CPU thread count, and timings.

- `comparison.png`：原始准确率与相对准确率对比图。

  `comparison.png`: comparison of accuracy and relative accuracy.

- `category_comparison.png`：四种抑制条件的逐大类准确率图；点为各类准确率，竖线为该模型在所有图片上的准确率。图中只含三个 CNN，没有 human 数据。

  `category_comparison.png`: per-category accuracy for the four suppression conditions. Dots show category accuracy; vertical lines show each model's overall accuracy in the corresponding condition. The plot includes the three CNNs and no human data.

已有 `predictions.csv` 和 `metrics.csv` 时，无需重新推理即可单独重画逐类图。macOS 终端运行：

If you already have `predictions.csv` and `metrics.csv`, you can redraw the per-category plot without running inference again. On macOS, run:

```bash
.venv/bin/python plot_categories.py results/demo_800
```

Windows PowerShell 运行：

In Windows PowerShell, run:

```powershell
.\.venv\Scripts\python.exe plot_categories.py results\demo_800
```

`--smoke` 输出到 `results/demo_16/`。这些结果和 `data/`、`.venv/` 都被 `.gitignore` 排除，不随 Git 仓库分发。Windows 系统的 `report.json` 中，`peak_process_rss_mib` 为 `null`，磁盘占用按文件的逻辑大小统计；macOS 则记录进程峰值 RSS 和实际分配的文件块大小。

The `--smoke` run writes to `results/demo_16/`. Output files, `data/`, and `.venv/` are excluded by `.gitignore` and are not distributed with the repository. On Windows, `peak_process_rss_mib` in `report.json` is `null` and disk usage is reported using logical file sizes. On macOS, the report records peak process RSS and disk usage based on allocated blocks.

## 复现范围与来源

**Reproduction scope and sources**

依赖版本在 `requirements-lock.txt` 固定。`config/model_weights.json` 固定三个 Hugging Face 仓库的提交修订号和 SHA-256；程序从该不可变修订号下载 `model.safetensors` 并核对哈希。`report.json` 同时记录本次使用的修订号与哈希。若远端移除这些提交，需选择仍可获取的权重、更新锁文件，并重新说明结果差异。

Package versions are pinned in `requirements-lock.txt`. The file `config/model_weights.json` pins a commit revision and SHA-256 checksum for each of the three Hugging Face repositories. The program downloads `model.safetensors` from the pinned revision and checks its hash. The revision and hash used for each run are also recorded in `report.json`. If a pinned revision is removed, select available weights, update the lock file, and document the change when comparing results.

实验方法见[论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/57a9b97477b67936298489e3c1417b0a-Paper-Conference.pdf)及[官方补充材料](https://proceedings.neurips.cc/paper_files/paper/2025/file/57a9b97477b67936298489e3c1417b0a-Supplemental-Conference.zip)。本仓库只实现实验 1 的推理与统计，不包含原论文的人工筛选图片、训练过程或原始 checkpoint。

See the [paper PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/57a9b97477b67936298489e3c1417b0a-Paper-Conference.pdf) and [official supplementary material](https://proceedings.neurips.cc/paper_files/paper/2025/file/57a9b97477b67936298489e3c1417b0a-Supplemental-Conference.zip) for the experimental method. This repository implements model inference and scoring for Experiment 1. It does not include the authors' curated image set, training workflow, or original checkpoints.

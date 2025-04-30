<p align="center">
  <img src="assets/gfpgan_logo.png" height=130>
</p>

## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

<div align="center">

[![download](https://img.shields.io/github/downloads/TencentARC/GFPGAN/total.svg)](https://github.com/TencentARC/GFPGAN/releases)
[![PyPI](https://img.shields.io/pypi/v/gfpgan)](https://pypi.org/project/gfpgan/)
[![Open issue](https://img.shields.io/github/issues/TencentARC/GFPGAN)](https://github.com/TencentARC/GFPGAN/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/TencentARC/GFPGAN)](https://github.com/TencentARC/GFPGAN/issues)
[![LICENSE](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE)
[![python lint](https://github.com/TencentARC/GFPGAN/actions/workflows/pylint.yml/badge.svg)](https://github.com/TencentARC/GFPGAN/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/TencentARC/GFPGAN/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/TencentARC/GFPGAN/blob/master/.github/workflows/publish-pip.yml)
</div>

1. :boom: **更新** 在线演示：[![Replicate](https://img.shields.io/static/v1?label=Demo&message=Replicate&color=blue)](https://replicate.com/tencentarc/gfpgan)。备用链接：[Replicate 备份](https://replicate.com/xinntao/gfpgan)。
1. :boom: **更新** 在线演示：[![Huggingface Gradio](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/Xintao/GFPGAN)
1. [Colab 演示](https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo) <a href="https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>；另一个 [Colab 演示](https://colab.research.google.com/drive/1Oa1WwKB4M4l1GmR7CtswDVgOCOeSLChA?usp=sharing) 使用论文中的原始模型。

> :rocket: **感谢你对我们工作的关注。你可能还会对我们在 [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_video_model.md) 项目中关于动漫图像和视频的微型模型更新感兴趣！** :blush:

GFPGAN 致力于开发一个**面向真实场景的人脸修复实用算法**。<br>
该方法利用预训练人脸 GAN（如 StyleGAN2）中丰富多样的先验信息进行盲人脸修复。

:question: 常见问题请查阅 [FAQ.md](FAQ.md)。

:triangular_flag_on_post: **更新日志**

- :white_check_mark: 添加 [RestoreFormer](https://github.com/wzhouxiff/RestoreFormer) 推理代码。
- :white_check_mark: 添加 [V1.4 模型](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth)，相较于 V1.3，细节更丰富，身份保持更好。
- :white_check_mark: 添加 **[V1.3 模型](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth)**，在*超低质量*或*高质量*输入图像上表现更好，结果更自然。详情请见 [模型列表](#european_castle-model-zoo) 与 [Comparisons.md](Comparisons.md)。
- :white_check_mark: 集成至 [Huggingface Spaces](https://huggingface.co/spaces)，采用 [Gradio](https://github.com/gradio-app/gradio) 构建前端。见 [Gradio 在线演示](https://huggingface.co/spaces/akhaliq/GFPGAN)。
- :white_check_mark: 支持使用 [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 增强非人脸区域（背景）。
- :white_check_mark: 提供无需 CUDA 扩展的 *纯净* 版本。
- :white_check_mark: 提供不进行人脸上色的新版模型。

---

如果 GFPGAN 对你的照片或项目有所帮助，请为本项目点个 :star:，或推荐给你的朋友。感谢支持！:blush:

推荐项目：<br>
:arrow_forward: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)：通用图像恢复的实用算法<br>
:arrow_forward: [BasicSR](https://github.com/xinntao/BasicSR)：开源图像/视频恢复工具箱<br>
:arrow_forward: [facexlib](https://github.com/xinntao/facexlib)：包含丰富人脸处理功能的工具集<br>
:arrow_forward: [HandyView](https://github.com/xinntao/HandyView)：基于 PyQt5 的轻量级图像浏览与对比工具<br>

---

### :book: GFP-GAN：基于生成式人脸先验的真实世界盲人脸修复

> [[论文](https://arxiv.org/abs/2101.04061)] &emsp; [[项目主页](https://xinntao.github.io/projects/gfpgan)] &emsp; [演示] <br>
> [Xintao Wang](https://xinntao.github.io/), [Yu Li](https://yu-li.github.io/), [Honglun Zhang](https://scholar.google.com/citations?hl=en&user=KjQLROoAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> 腾讯 PCG 应用研究中心（ARC）

<p align="center">
  <img src="https://xinntao.github.io/projects/GFPGAN_src/gfpgan_teaser.jpg">
</p>

---

## :wrench: 环境依赖与安装

- Python >= 3.7（推荐使用 [Anaconda](https://www.anaconda.com/download/#linux) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)）
- [PyTorch >= 1.7](https://pytorch.org/)
- 可选：NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- 可选：Linux 系统

### 安装步骤

我们现在提供了无需 CUDA 扩展的 *纯净* 版本。<br>
若需使用论文中原始模型，请参阅 [PaperModel.md](PaperModel.md)。

1. 克隆项目

    ```bash
    git clone https://github.com/TencentARC/GFPGAN.git
    cd GFPGAN
    ```

1. 安装依赖包

    ```bash
    # 安装 basicsr - https://github.com/xinntao/BasicSR
    pip install basicsr

    # 安装 facexlib - https://github.com/xinntao/facexlib
    pip install facexlib

    pip install -r requirements.txt
    python setup.py develop

    # 若希望增强背景（非人脸区域），需安装 realesrgan 包
    pip install realesrgan
    ```

## :zap: 快速使用

我们以 v1.3 版本为例。更多模型请见 [模型列表](#european_castle-model-zoo)。

下载预训练模型：[GFPGANv1.3.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth)

```bash
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models
```

**运行推理**

```bash
python inference_gfpgan.py -i inputs/whole_imgs -o results -v 1.3 -s 2
```
```console
使用方法: python inference_gfpgan.py -i inputs/whole_imgs -o results -v 1.3 -s 2 [可选项]...

  -h                   显示帮助信息
  -i input             输入图像或文件夹。默认：inputs/whole_imgs
  -o output            输出文件夹。默认：results
  -v version           GFPGAN 模型版本。选项：1 | 1.2 | 1.3。默认：1.3
  -s upscale           图像最终上采样倍数。默认：2
  -bg_upsampler        背景增强器。默认：realesrgan
  -bg_tile             背景增强时的 tile 大小，0 表示不分块测试。默认：400
  -suffix              修复人脸的文件名后缀
  -only_center_face    仅修复中心人脸
  -aligned             输入图像是否为已对齐人脸
  -ext                 图像扩展名。选项：auto | jpg | png；auto 表示与输入相同。默认：auto
```
若需使用论文原始模型，请参考 [PaperModel.md](PaperModel.md) 了解安装与使用方法。

## :european_castle: 模型列表（Model Zoo）
| 版本 | 模型名称 | 模型说明 |
| :---: | :--- | :--- |
| V1.3 | [GFPGANv1.3.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth) | 基于 V1.2；恢复效果更自然；在极低质量与高质量图像中表现更佳。 |
| V1.2 | [GFPGANCleanv1-NoCE-C2.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth) | 无人脸上色；无需 CUDA 扩展；使用更多预处理数据训练。 |
| V1   | [GFPGANv1.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth) | 论文中使用的模型，包含上色模块。 |

模型对比效果请参考[Comparisons.md](Comparisons.md)

请注意，V1.3 不一定始终优于 V1.2，建议根据图像情况自行选择。

| 版本 | 优点 | 缺点 |
| :---: | :---: | :---: |
| V1.3 | ✓ 输出更加自然 <br> ✓ 对极低质量输入效果更好 <br> ✓ 对较高质量输入也有效 <br> ✓ 支持重复修复（两次） | ✗ 图像不够锐利 <br> ✗ 身份可能发生轻微变化 |
| V1.2 | ✓ 输出更锐利 <br> ✓ 带有美颜化妆效果 | ✗ 某些输出看起来不自然 |

你可以在以下位置找到**更多模型（如判别器discriminators）**：  
[[Google Drive](https://drive.google.com/drive/folders/17rLiFzcUMoQuhLnptDsKolegHWwJOnHu?usp=sharing)]，或 [[腾讯微云](https://share.weiyun.com/ShYoCCoc)]

---

## :computer: 训练

我们提供了用于 GFPGAN 的训练代码（与论文中使用一致）。<br>你可以根据自己的需求对其进行改进。

**小贴士：**

1. 更多高质量人脸样本有助于提升修复效果；
2. 你可能需要进行一些预处理，如美颜化妆等。

**训练流程：**

（你可以尝试一个简化版本：`options/train_gfpgan_v1_simple.yml`，该版本不需要人脸部件关键点）

1. 准备数据集：[FFHQ](https://github.com/NVlabs/ffhq-dataset)

2. 下载预训练模型和其他依赖数据，并放入 `experiments/pretrained_models` 文件夹：
    - [StyleGAN2 预训练模型：StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth)
    - [FFHQ 的眼睛和嘴巴关键点位置信息：FFHQ_eye_mouth_landmarks_512.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/FFHQ_eye_mouth_landmarks_512.pth)
    - [简化版 ArcFace 模型：arcface_resnet18.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/arcface_resnet18.pth)

3. 根据需要修改配置文件 `options/train_gfpgan_v1.yml`

4. 开始训练：

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=22021 gfpgan/train.py -opt options/train_gfpgan_v1.yml --launcher pytorch
```

---

## :scroll: 许可证与致谢

GFPGAN 遵循 Apache License 2.0 开源协议。

---

## BibTeX 引用格式

```bibtex
@InProceedings{wang2021gfpgan,
    author = {Xintao Wang and Yu Li and Honglun Zhang and Ying Shan},
    title = {Towards Real-World Blind Face Restoration with Generative Facial Prior},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2021}
}
```

---

## :e-mail: 联系方式

如有任何问题，请通过以下邮箱联系：

- `xintao.wang@outlook.com`
- `xintaowang@tencent.com`

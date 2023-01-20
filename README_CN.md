<p align="center">
  <img src="assets/gfpgan_logo.png" height=130>
</p>
## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

<div align="center">
<!-- <a href="https://twitter.com/_Xintao_" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/17445847/187162058-c764ced6-952f-404b-ac85-ba95cce18e7b.png" width="4%" alt="" />
</a> -->


[![download](https://img.shields.io/github/downloads/TencentARC/GFPGAN/total.svg)](https://github.com/TencentARC/GFPGAN/releases)
[![PyPI](https://img.shields.io/pypi/v/gfpgan)](https://pypi.org/project/gfpgan/)
[![Open issue](https://img.shields.io/github/issues/TencentARC/GFPGAN)](https://github.com/TencentARC/GFPGAN/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/TencentARC/GFPGAN)](https://github.com/TencentARC/GFPGAN/issues)
[![LICENSE](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE)
[![python lint](https://github.com/TencentARC/GFPGAN/actions/workflows/pylint.yml/badge.svg)](https://github.com/TencentARC/GFPGAN/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/TencentARC/GFPGAN/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/TencentARC/GFPGAN/blob/master/.github/workflows/publish-pip.yml)

</div>

- 以下是对英文版的翻译，个人水平有限，如有错误欢迎指正。– Translated by **[@FreemanRyan](https://github.com/FreemanRyan)**


1. :boom: **更新** 在线 示例: [![Replicate](https://img.shields.io/static/v1?label=Demo&message=Replicate&color=blue)](https://replicate.com/tencentarc/gfpgan). 这是 [备份](https://replicate.com/xinntao/gfpgan).
1. :boom: **更新** 在线 示例: [![Huggingface Gradio](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/Xintao/GFPGAN)
1. [Colab Demo](https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo) for GFPGAN <a href="https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>; (原始论文的另一个模型展示 [Colab Demo](https://colab.research.google.com/drive/1Oa1WwKB4M4l1GmR7CtswDVgOCOeSLChA?usp=sharing) )

<!-- 3. 在线 示例: [Replicate.ai](https://replicate.com/xinntao/gfpgan) (可能需要登录, 将返回显示完整图片)

4. 在线 示例: [Baseten.co](https://app.baseten.co/applications/Q04Lz0d/operator_views/8qZG6Bg) (由GPU提供支持, 将返回显示完整图片)
5. 我们提供一个纯净的GFPGAN版本, 不包含CUDA 扩展. 因此它可以在**Windows** 环境或者**CPU mode**模式下运行. -->

> :rocket: **感谢您对我们的工作感兴趣. 您或许对我们在轻量化模型中进行动漫图片及视频处理感兴趣，右侧为链接可点击查看。 [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_video_model.md)** :blush:

GFPGAN 致力于开发一个**真实世界人脸修复的实用算法**.<br>

它使用封装在GAN (*e.g.*, StyleGAN2) 中丰富多样的预训练的人脸模型来对模糊人脸进行AI修复.

:question: 常见的问答可点击右侧链接查看 [FAQ.md](FAQ.md).

:triangular_flag_on_post: **更新**

- :white_check_mark: 添加 [RestoreFormer](https://github.com/wzhouxiff/RestoreFormer) 推理代码(用来加载训练好的模型并对新的数据进行预测).
- :white_check_mark: 添加 [V1.4 model](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth), 比 V1.3 产生更多细节和更好的特征.
- :white_check_mark: 添加 **[V1.3 model](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth)**  对与处理极低/极高图片质量，处理结果将更自然质量更高, . 详见 [Model zoo](#european_castle-model-zoo), [Comparisons.md](Comparisons.md)
- :white_check_mark: 使用[Gradio](https://github.com/gradio-app/gradio) 集成到[Huggingface Spaces](https://huggingface.co/spaces) . 详见[Gradio Web Demo](https://huggingface.co/spaces/akhaliq/GFPGAN).
- :white_check_mark: 使用 [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 支持增强对非人脸（背景）区域的处理.
- :white_check_mark: 我们提供一个纯净的GFPGAN版本, 不要求包含CUDA 扩展. 
- :white_check_mark: 我们提供一个非彩色人脸更新模型.

---

如果GFPGAN 对您的照片或者项目有帮助, 欢迎您收藏:star:这个git仓库或者向您的朋友推荐它。 非常感谢。:blush:
以下为您推荐一些其他相关的项目:<br>
:arrow_forward: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): 通用图像复原的实用算法<br>
:arrow_forward: [BasicSR](https://github.com/xinntao/BasicSR): 开源图像和视频恢复工具箱<br>
:arrow_forward: [facexlib](https://github.com/xinntao/facexlib): 提供有用的人脸关系功能的集合<br>
:arrow_forward: [HandyView](https://github.com/xinntao/HandyView): 一个基于PyQt5的图像查看器，便于查看和比较<br>

---

### :book: GFP-GAN: 通过人脸预测模型实现真实世界的模糊面部修复

> [[Paper](https://arxiv.org/abs/2101.04061)] &emsp; [[Project Page](https://xinntao.github.io/projects/gfpgan)] &emsp; [Demo] <br>
> [Xintao Wang](https://xinntao.github.io/), [Yu Li](https://yu-li.github.io/), [Honglun Zhang](https://scholar.google.com/citations?hl=en&user=KjQLROoAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> Applied Research Center (ARC), Tencent PCG

<p align="center">
  <img src="https://xinntao.github.io/projects/GFPGAN_src/gfpgan_teaser.jpg">
</p>

---

## :wrench: 依赖和安装

- Python >= 3.7 (推荐使用 [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- 可选: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- 可选: Linux

### 安装

我们提供一个纯净的GFPGAN版本, 不要求包含个性化的CUDA 扩展. <br>
如果你想使用我们论文中的原始模型, 请查阅 [PaperModel.md](PaperModel.md) 中的安装步骤.

1. 克隆仓库

    ```bash
    git clone https://github.com/TencentARC/GFPGAN.git
    cd GFPGAN
    ```

1. 安装依赖包

    ```bash
    # Install basicsr - https://github.com/xinntao/BasicSR
    # 我们使用 BasicSR 进行训练和预测
    pip install basicsr

    # Install facexlib - https://github.com/xinntao/facexlib
    # 我们使用facexlib包中的人脸检测和人脸修复助手
    pip install facexlib

    pip install -r requirements.txt
    python setup.py develop

    # 如果您想使用 Real-ESRGAN 增强背景（非面部）区域,
    # 您还需要安装 realesrgan 包
    pip install realesrgan
    ```

## :zap: 快速预测

我们使用 v1.3 版本作为例子. 更多模型可以在右侧链接中找到 [here](#european_castle-model-zoo).

点击右侧链接，下载预训练模型: [GFPGANv1.3.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth)

```bash
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models
```

**使用以下命令进行预测修复!**

```bash
python inference_gfpgan.py -i inputs/whole_imgs -o results -v 1.3 -s 2
```

```console
Usage: python inference_gfpgan.py -i inputs/whole_imgs -o results -v 1.3 -s 2 [options]...

  -h                   查阅帮助
  -i input             打开照片或者文件夹. Default: inputs/whole_imgs
  -o output            保存路径. Default: results
  -v version           GFPGAN 模型版本. Option: 1 | 1.2 | 1.3. Default: 1.3
  -s upscale           图像的最终升采样比例. Default: 2
  -bg_upsampler        背景升采样器. Default: realesrgan
  -bg_tile             背景采样器的系数大小, 0表示测试期间不使用系数. Default: 400
  -suffix              输出结果后缀
  -only_center_face    只对中部人脸进行修复
  -aligned             输入是整齐人脸
  -ext                 图片扩展名. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
```

如果你想使用我们论文中的原始模型, 请查阅 [PaperModel.md](PaperModel.md) 中的安装步骤以便使用.

## :european_castle: 已有模型库

| Version |                          Model Name                          |                       Description                       |
| :-----: | :----------------------------------------------------------: | :-----------------------------------------------------: |
|  V1.3   | [GFPGANv1.3.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth) |          基于 V1.2; 更自然且高质量的修复结果.           |
|  V1.2   | [GFPGANCleanv1-NoCE-C2.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth) | 黑白; 无需 CUDA 扩展. 通过预处理使用更多数据进行训练。. |
|   V1    | [GFPGANv1.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth) |                     论文模型, 彩色.                     |

具体差异详见 [Comparisons.md](Comparisons.md).

请注意，V1.3 并不总是优于 V1.2。 您可能需要根据您的目的和输入选择不同的模型。

| 版本 |                             优点                             |                缺点                 |
| :--: | :----------------------------------------------------------: | :---------------------------------: |
| V1.3 | ✓ 更自然<br> ✓对低质量的照片有更好的修复效果 <br> ✓ 对高质量图片有更好的细节把控 <br>✓ 能够进行二次修复 | ✗不够清晰 <br> ✗ 会改变一些特征细节 |
| V1.2 |               ✓ 更清晰的输出 <br> ✓ 有美颜效果               |       ✗ 一些修复结果不够自然        |

这里有 **更多模型 (例如 鉴别器)** : [[Google Drive](https://drive.google.com/drive/folders/17rLiFzcUMoQuhLnptDsKolegHWwJOnHu?usp=sharing)], OR [[Tencent Cloud 腾讯微云](https://share.weiyun.com/ShYoCCoc)]

## :computer: 训练

我们提供了 GFPGAN 的训练代码（已包含在我们的论文中）。 <br>
您可以根据自己的需要对其进行改进。

**友情提示**

1. 更多高质量的人脸可以提高修复质量.
2. 可能需要进行一些预处理，比如美颜.

**程序**

(您可以先尝试一个简单的不包含面部识别的组件版本 ( `options/train_gfpgan_v1_simple.yml`).)

1. 数据集准备: [FFHQ](https://github.com/NVlabs/ffhq-dataset)

1. 下载预训练模型及其他数据. 将它们放到 `experiments/pretrained_models` 文件夹中.
    1. [Pre-trained StyleGAN2 model: StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth)
    1. [Component locations of FFHQ: FFHQ_eye_mouth_landmarks_512.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/FFHQ_eye_mouth_landmarks_512.pth)
    1. [A simple ArcFace model: arcface_resnet18.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/arcface_resnet18.pth)

1. 修改相应的配置文件 `options/train_gfpgan_v1.yml` .

1. 执行训练脚本

> python -m torch.distributed.launch --nproc_per_node=4 --master_port=22021 gfpgan/train.py -opt options/train_gfpgan_v1.yml --launcher pytorch

## :scroll: 许可证书及认证

GFPGAN 在 Apache License Version 2.0 下发布。

## BibTeX 目录生成

    @InProceedings{wang2021gfpgan,
        author = {Xintao Wang and Yu Li and Honglun Zhang and Ying Shan},
        title = {Towards Real-World Blind Face Restoration with Generative Facial Prior},
        booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2021}
    }

## :e-mail: 联系方式

如果您有任何疑问， 请发送邮件至 `xintao.wang@outlook.com` or `xintaowang@tencent.com`.
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

1. :boom: **已更新** 在线演示: [![Replicate](https://img.shields.io/static/v1?label=Demo&message=Replicate&color=blue)](https://replicate.com/tencentarc/gfpgan). [备份链接](https://replicate.com/xinntao/gfpgan).
2. :boom: **已更新** 在线演示: [![Huggingface Gradio](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/Xintao/GFPGAN)
3. [Colab演示](https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo)<a href="https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>; (另一个根据原版论文模型实现的[Colab演示](https://colab.research.google.com/drive/1Oa1WwKB4M4l1GmR7CtswDVgOCOeSLChA?usp=sharing) )

<!-- 3. Online demo: [Replicate.ai](https://replicate.com/xinntao/gfpgan) (may need to sign in, return the whole image)
4. Online demo: [Baseten.co](https://app.baseten.co/applications/Q04Lz0d/operator_views/8qZG6Bg) (backed by GPU, returns the whole image)
5. We provide a *clean* version of GFPGAN, which can run without CUDA extensions. So that it can run in **Windows** or on **CPU mode**. -->

> :rocket: **感谢您对我们工作的兴趣. 也许您想了解一下我们最新的基于*小模型*的*动画图片与视频*[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_video_model.md)模型** :blush:

GFPGAN旨在开发一个**针对现实世界中面部修复的实用算法**.<br>
本算法利用丰富多样的先验数据，封装在一个预训练的人脸GAN（*例如* StyleGAN2）网络中，进行盲面部修复.

:question: 常见问题请参考[FAQ.md](FAQ.md).

:triangular_flag_on_post: **更新日志**

- :white_check_mark: 增加[RestoreFormer](https://github.com/wzhouxiff/RestoreFormer)推理代码.
- :white_check_mark: 增加[V1.4模型](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth)，相比V1.3模型会生成更多细节并提高辨识度.
- :white_check_mark: 增加[V1.3模型](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth), 此版本生成更**自然**的修复结果，并在*极低质量*/*高质量*的输入时生成更好的效果.详情移步至[模型库](#european_castle-model-zoo), [对比](Comparisons.md)
- :white_check_mark: 集成至[Huggingface Spaces](https://huggingface.co/spaces)和[Gradio](https://github.com/gradio-app/gradio). 查看[Gradio Web Demo](https://huggingface.co/spaces/akhaliq/GFPGAN).
- :white_check_mark: 支持增强非面部区域（背景）通过[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
- :white_check_mark: 我们提供一个*纯净*版GFPGAN，运行无需CUDA.
- :white_check_mark: 我们提供了一个更新的模型，不对人脸进行着色.

---
如果GFPGAN对您的照片/项目有帮助, 请帮忙:star:本项目或推荐给您的朋友们, 感谢:blush:
其他推荐项目:<br>
:arrow_forward: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): 一种实用的通用图像修复算法<br>
:arrow_forward: [BasicSR](https://github.com/xinntao/BasicSR): 一个开源的图像和视频修复工具箱<br>
:arrow_forward: [facexlib](https://github.com/xinntao/facexlib): 一个提供面部相关函数的集合库<br>
:arrow_forward: [HandyView](https://github.com/xinntao/HandyView): 一个基于PyQt5的图片查看器, 方便查看和比较<br>

---

### :book: GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior

> [[论文](https://arxiv.org/abs/2101.04061)] &emsp; [[项目主页](https://xinntao.github.io/projects/gfpgan)] &emsp; [演示] <br>
> [Xintao Wang](https://xinntao.github.io/), [Yu Li](https://yu-li.github.io/), [Honglun Zhang](https://scholar.google.com/citations?hl=en&user=KjQLROoAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> Applied Research Center (ARC), Tencent PCG

<p align="center">
  <img src="https://xinntao.github.io/projects/GFPGAN_src/gfpgan_teaser.jpg">
</p>

---

## :wrench: 依赖与安装

- Python >= 3.7 (推荐使用[Anaconda](https://www.anaconda.com/download/#linux)或[Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- 可选: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- 可选: Linux

### 安装

我们现在提供一个*纯净*版GFPGAN, 运行无需CUDA. <br>
如果您想使用我们论文的原版模型, 请参考[论文模型](PaperModel.md)安装.

1. 克隆代码

   ```bash
   git clone https://github.com/TencentARC/GFPGAN.git
   cd GFPGAN
   ```

2. 安装依赖包

   ```bash
   # 安装 basicsr - https://github.com/xinntao/BasicSR
   # 我们使用BaiscSR进行训练和推理
   pip install basicsr

   # 安装 facexlib - https://github.com/xinntao/facexlib
   # 我们使用facexlib软件包中的面部检测和面部修复辅助工具
   pip install facexlib

   pip install -r requirements.txt
   python setup.py develop

   # 如果您想使用Real-ESRGAN增强背景（非面部）区域
   # 您需要安装 realesrgan 包
   pip install realesrgan
   ```

## :zap: 快速推理

我们以V1.3模型为例. 更多模型可以查看[此处](#european_castle-model-zoo).

下载预训练模型: [GFPGANv1.3.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth)

```bash
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models
```

**推理!**

```bash
python inference_gfpgan.py -i inputs/whole_imgs -o results -v 1.3 -s 2
```

```console
Usage: python inference_gfpgan.py -i inputs/whole_imgs -o results -v 1.3 -s 2 [options]...

  -h                   显示此帮助
  -i input             输入图片或目录. 默认: inputs/whole_imgs
  -o output            输出目录. 默认: results
  -v version           GFPGAN模型版本. 可选: 1 | 1.2 | 1.3. 默认: 1.3
  -s upscale           图像的最终超采样比例. 默认: 2
  -bg_upsampler        背景超采样方式. 默认: realesrgan
  -bg_tile             背景采样器的片尺寸, 0表示无. 默认: 400
  -suffix              输出文件的后缀
  -only_center_face    仅修复中心面部
  -aligned             输入是对齐的面孔
  -ext                 输入图像类型. 可选: auto | jpg | png, auto 意味着使用相同类型作为输入, 默认: auto
```

如果您想使用我们论文中的原版模型, 请参考[论文模型](PaperModel.md)进行安装和推理.

## :european_castle: 模型库

| 版本 | 模型名称                                                                                                           | 描述                                                                                                    |
|:-------:|:--------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|
| V1.3    | [GFPGANv1.3.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth)                       | 基于V1.2; **更自然**的修复结果; 在*极低质量*/*高质量*的输入时生成更好的效果. |
| V1.2    | [GFPGANCleanv1-NoCE-C2.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth) | 无上色;无必须CUDA支持. 通过更多预处理后的数据进行训练.                  |
| V1      | [GFPGANv1.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth)                           | 论文原版模型, 有上色.                                                                            |

对比请参考[此处](Comparisons.md).

请注意V1.3并不是总比V1.2效果好. 您也许需要根据您的需求和输入而选择不同的模型.

| 版本 | 优势                                                                                                                                                   | 劣势                                               |
|:-------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------:|
| V1.3    | ✓ 自然的输出<br> ✓在低质量输入时有更好的效果 <br> ✓ 可在相对高质量输入时工作 <br>✓ 可以重复（两次）进行修复 | ✗ 不是非常清晰 <br> ✗ 对面容有明显改变 |
| V1.2    | ✓ 更清晰的输出 <br> ✓ 自带美颜效果                                                                                                                  | ✗ 一些输出非常不自然                             |

您可以在下面链接找到**更多模型(例如discriminators)**: [[Google Drive](https://drive.google.com/drive/folders/17rLiFzcUMoQuhLnptDsKolegHWwJOnHu?usp=sharing)], 或[[腾讯微云](https://share.weiyun.com/ShYoCCoc)]

## :computer: 训练

我们提供GFPGAN的训练代码(在论文中所使用的). <br>
您可以根据您的需求自行调节.

**提示**

1. 高质量的面部输入可以提高修复质量.
2. 您也许需要一些预处理, 例如美颜.

**流程**

(您可以尝试一个简单的版本( `options/train_gfpgan_v1_simple.yml`)并不需要面部标记.)

1. 准备数据集: [FFHQ](https://github.com/NVlabs/ffhq-dataset)

2. 下载预训练的模型和其他数据. 把它们放在`experiments/pretrained_models`目录.

   1. [Pre-trained StyleGAN2 model: StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth)
   2. [Component locations of FFHQ: FFHQ_eye_mouth_landmarks_512.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/FFHQ_eye_mouth_landmarks_512.pth)
   3. [A simple ArcFace model: arcface_resnet18.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/arcface_resnet18.pth)

3. 对配置文件`options/train_gfpgan_v1.yml`进行相应修改.

4. 训练

> python -m torch.distributed.launch --nproc_per_node=4 --master_port=22021 gfpgan/train.py -opt options/train_gfpgan_v1.yml --launcher pytorch

## :scroll: 版权与致谢

GFPGAN以Apache License Version 2.0许可证发布.

## BibTeX

    @InProceedings{wang2021gfpgan,
        author = {Xintao Wang and Yu Li and Honglun Zhang and Ying Shan},
        title = {Towards Real-World Blind Face Restoration with Generative Facial Prior},
        booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2021}
    }

## :e-mail: 联系方式

如果您有任何疑问, 请邮件联系 `xintao.wang@outlook.com` 或 `xintaowang@tencent.com`.

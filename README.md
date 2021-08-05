# GFPGAN (CVPR 2021)

[![download](https://img.shields.io/github/downloads/TencentARC/GFPGAN/total.svg)](https://github.com/TencentARC/GFPGAN/releases)
[![Open issue](https://isitmaintained.com/badge/open/TencentARC/GFPGAN.svg)](https://github.com/TencentARC/GFPGAN/issues)
[![LICENSE](https://img.shields.io/github/license/TencentARC/GFPGAN.svg)](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE)
[![python lint](https://github.com/TencentARC/GFPGAN/actions/workflows/pylint.yml/badge.svg)](https://github.com/TencentARC/GFPGAN/blob/master/.github/workflows/pylint.yml)

[**Paper**](https://arxiv.org/abs/2101.04061) **|** [**Project Page**](https://xinntao.github.io/projects/gfpgan) &emsp;&emsp; [English](README.md) **|** [简体中文](README_CN.md)

GFPGAN is a blind face restoration algorithm towards real-world face images.

<a href="https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>
[Colab Demo](https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo)

### :book: GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior
> [[Paper](https://arxiv.org/abs/2101.04061)] &emsp; [[Project Page](https://xinntao.github.io/projects/gfpgan)] &emsp; [Demo] <br>
> [Xintao Wang](https://xinntao.github.io/), [Yu Li](https://yu-li.github.io/), [Honglun Zhang](https://scholar.google.com/citations?hl=en&user=KjQLROoAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> Applied Research Center (ARC), Tencent PCG

#### Abstract

Blind face restoration usually relies on facial priors, such as facial geometry prior or reference prior, to restore realistic and faithful details. However, very low-quality inputs cannot offer accurate geometric prior while high-quality references are inaccessible, limiting the applicability in real-world scenarios. In this work, we propose GFP-GAN that leverages **rich and diverse priors encapsulated in a pretrained face GAN** for blind face restoration. This Generative Facial Prior (GFP) is incorporated into the face restoration process via novel channel-split spatial feature transform layers, which allow our method to achieve a good balance of realness and fidelity. Thanks to the powerful generative facial prior and delicate designs, our GFP-GAN could jointly restore facial details and enhance colors with just a single forward pass, while GAN inversion methods require expensive image-specific optimization at inference. Extensive experiments show that our method achieves superior performance to prior art on both synthetic and real-world datasets.

#### BibTeX

    @InProceedings{wang2021gfpgan,
        author = {Xintao Wang and Yu Li and Honglun Zhang and Ying Shan},
        title = {Towards Real-World Blind Face Restoration with Generative Facial Prior},
        booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2021}
    }

<p align="center">
  <img src="https://xinntao.github.io/projects/GFPGAN_src/gfpgan_teaser.jpg">
</p>

---

## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Linux (We have not tested on Windows)

### Installation

1. Clone repo

    ```bash
    git clone https://github.com/xinntao/GFPGAN.git
    cd GFPGAN
    ```

1. Install dependent packages

    As StyleGAN2 uses customized PyTorch C++ extensions, you need to **compile them during installation** or **load then just-in-time(JIT)**.
    You can refer to [BasicSR-INSTALL.md](https://github.com/xinntao/BasicSR/blob/master/INSTALL.md) for more details.

    **Option 1: Load extensions just-in-time(JIT)** (For those just want to do simple inferences, may have less issues)

    ```bash
    # Install basicsr - https://github.com/xinntao/BasicSR
    # We use BasicSR for both training and inference
    pip install basicsr

    # Install facexlib - https://github.com/xinntao/facexlib
    # We use face detection and face restoration helper in the facexlib package
    pip install facexlib

    pip install -r requirements.txt

    # remember to set BASICSR_JIT=True before your running commands
    ```

    **Option 2: Compile extensions during installation** (For those need to train/inference for many times)

    ```bash
    # Install basicsr - https://github.com/xinntao/BasicSR
    # We use BasicSR for both training and inference
    # Set BASICSR_EXT=True to compile the cuda extensions in the BasicSR - It may take several minutes to compile, please be patient
    # Add -vvv for detailed log prints
    BASICSR_EXT=True pip install basicsr -vvv

    # Install facexlib - https://github.com/xinntao/facexlib
    # We use face detection and face restoration helper in the facexlib package
    pip install facexlib

    pip install -r requirements.txt
    ```

## :zap: Quick Inference

Download pre-trained models: [GFPGANv1.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth)

```bash
wget https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth -P experiments/pretrained_models
```

- Option 1: Load extensions just-in-time(JIT)

    ```bash
    BASICSR_JIT=True python inference_gfpgan_full.py --model_path experiments/pretrained_models/GFPGANv1.pth --test_path inputs/whole_imgs

    # for aligned images
    BASICSR_JIT=True python inference_gfpgan_full.py --model_path experiments/pretrained_models/GFPGANv1.pth --test_path inputs/cropped_faces --aligned
    ```

- Option 2: Have successfully compiled extensions during installation

    ```bash
    python inference_gfpgan_full.py --model_path experiments/pretrained_models/GFPGANv1.pth --test_path inputs/whole_imgs

    # for aligned images
    python inference_gfpgan_full.py --model_path experiments/pretrained_models/GFPGANv1.pth --test_path inputs/cropped_faces --aligned
    ```

## :computer: Training

We provide complete training codes for GFPGAN. <br>
You could improve it according to your own needs.

1. Dataset preparation: [FFHQ](https://github.com/NVlabs/ffhq-dataset)

1. Download pre-trained models and other data. Put them in the `experiments/pretrained_models` folder.
    1. [Pretrained StyleGAN2 model: StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth)
    1. [Component locations of FFHQ: FFHQ_eye_mouth_landmarks_512.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/FFHQ_eye_mouth_landmarks_512.pth)
    1. [A simple ArcFace model: arcface_resnet18.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/arcface_resnet18.pth)

1. Modify the configuration file `train_gfpgan_v1.yml` accordingly.

1. Training

> python -m torch.distributed.launch --nproc_per_node=4 --master_port=22021 train.py -opt train_gfpgan_v1.yml --launcher pytorch

or load extensions just-in-time(JIT)

> BASICSR_JIT=True python -m torch.distributed.launch --nproc_per_node=4 --master_port=22021 train.py -opt train_gfpgan_v1.yml --launcher pytorch

## :scroll: License and Acknowledgement

GFPGAN is realeased under Apache License Version 2.0.

## :e-mail: Contact

If you have any question, please email `xintao.wang@outlook.com` or `xintaowang@tencent.com`.

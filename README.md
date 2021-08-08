# GFPGAN (CVPR 2021)

[![download](https://img.shields.io/github/downloads/TencentARC/GFPGAN/total.svg)](https://github.com/TencentARC/GFPGAN/releases)
[![PyPI](https://img.shields.io/pypi/v/gfpgan)](https://pypi.org/project/gfpgan/)
[![Open issue](https://isitmaintained.com/badge/open/TencentARC/GFPGAN.svg)](https://github.com/TencentARC/GFPGAN/issues)
[![LICENSE](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE)
[![python lint](https://github.com/TencentARC/GFPGAN/actions/workflows/pylint.yml/badge.svg)](https://github.com/TencentARC/GFPGAN/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/TencentARC/GFPGAN/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/TencentARC/GFPGAN/blob/master/.github/workflows/publish-pip.yml)

1. [Colab Demo](https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo) for GFPGAN <a href="https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>; (Another [Colab Demo](https://colab.research.google.com/drive/1Oa1WwKB4M4l1GmR7CtswDVgOCOeSLChA?usp=sharing) for the original paper model)
1. We provide a *clean* version of GFPGAN, which can run without CUDA extensions. So that it can run in **Windows** or on **CPU mode**.

GFPGAN aims at developing **Practical Algorithm for Real-world Face Restoration**.<br>
It leverages rich and diverse priors encapsulated in a pretrained face GAN (*e.g.*, StyleGAN2) for blind face restoration.

:triangular_flag_on_post: **Updates**

- :white_check_mark: We provide a *clean* version of GFPGAN, which does not require CUDA extensions.
- :white_check_mark: We provide an updated model without colorizing faces.

### :book: GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior

> [[Paper](https://arxiv.org/abs/2101.04061)] &emsp; [[Project Page](https://xinntao.github.io/projects/gfpgan)] &emsp; [Demo] <br>
> [Xintao Wang](https://xinntao.github.io/), [Yu Li](https://yu-li.github.io/), [Honglun Zhang](https://scholar.google.com/citations?hl=en&user=KjQLROoAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> Applied Research Center (ARC), Tencent PCG

<p align="center">
  <img src="https://xinntao.github.io/projects/GFPGAN_src/gfpgan_teaser.jpg">
</p>

---

## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Option: Linux (We have not tested on Windows)

### Installation

We now provide a *clean* version of GFPGAN, which does not require customized CUDA extensions. <br>
If you want want to use the original model in our paper, please see [PaperModel.md](PaperModel.md) for installation.

1. Clone repo

    ```bash
    git clone https://github.com/TencentARC/GFPGAN.git
    cd GFPGAN
    ```

1. Install dependent packages

    ```bash
    # Install basicsr - https://github.com/xinntao/BasicSR
    # We use BasicSR for both training and inference
    pip install basicsr

    # Install facexlib - https://github.com/xinntao/facexlib
    # We use face detection and face restoration helper in the facexlib package
    pip install facexlib

    pip install -r requirements.txt
    python setup.py develop
    ```

## :zap: Quick Inference

Download pre-trained models: [GFPGANCleanv1-NoCE-C2.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth)

```bash
wget https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth -P experiments/pretrained_models
```

**Inference!**

```bash
python inference_gfpgan.py --upscale_factor 2 --test_path inputs/whole_imgs --save_root results
```

## :european_castle: Model Zoo

- [GFPGANCleanv1-NoCE-C2.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth): No colorization; no CUDA extensions are required. It is still in training. Trained with more data with pre-processing.
- [GFPGANv1.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth): The paper model, with colorization.

## :computer: Training

We provide the training codes for GFPGAN (used in our paper). <br>
You could improve it according to your own needs.

**Tips**

1. More high quality faces can improve the restoration quality.
2. You may need to perform some pre-processing, such as beauty makeup.

**Procedures**

(You can try a simple version ( `options/train_gfpgan_v1_simple.yml`) that does not require face component landmarks.)

1. Dataset preparation: [FFHQ](https://github.com/NVlabs/ffhq-dataset)

1. Download pre-trained models and other data. Put them in the `experiments/pretrained_models` folder.
    1. [Pretrained StyleGAN2 model: StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth)
    1. [Component locations of FFHQ: FFHQ_eye_mouth_landmarks_512.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/FFHQ_eye_mouth_landmarks_512.pth)
    1. [A simple ArcFace model: arcface_resnet18.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/arcface_resnet18.pth)

1. Modify the configuration file `options/train_gfpgan_v1.yml` accordingly.

1. Training

> python -m torch.distributed.launch --nproc_per_node=4 --master_port=22021 gfpgan/train.py -opt options/train_gfpgan_v1.yml --launcher pytorch

## :scroll: License and Acknowledgement

GFPGAN is released under Apache License Version 2.0.

## BibTeX

    @InProceedings{wang2021gfpgan,
        author = {Xintao Wang and Yu Li and Honglun Zhang and Ying Shan},
        title = {Towards Real-World Blind Face Restoration with Generative Facial Prior},
        booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2021}
    }

## :e-mail: Contact

If you have any question, please email `xintao.wang@outlook.com` or `xintaowang@tencent.com`.

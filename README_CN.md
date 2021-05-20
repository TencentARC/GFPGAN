# GFPGAN (CVPR 2021)

[**Paper**](https://arxiv.org/abs/2101.04061) **|** [**Project Page**](https://xinntao.github.io/projects/gfpgan) &emsp;&emsp; [English](README.md) **|** [简体中文](README_CN.md)

GFPGAN is a blind face restoration algorithm towards real-world face images.

### GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior

[Xintao Wang](https://xinntao.github.io/), [Yu Li](https://yu-li.github.io/), [Honglun Zhang](https://scholar.google.com/citations?hl=en&user=KjQLROoAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
Applied Research Center (ARC), Tencent PCG

#### Abstract

Blind face restoration usually relies on facial priors, such as facial geometry prior or reference prior, to restore realistic and faithful details. However, very low-quality inputs cannot offer accurate geometric prior while high-quality references are inaccessible, limiting the applicability in real-world scenarios. In this work, we propose GFP-GAN that leverages rich and diverse priors encapsulated in a pretrained face GAN for blind face restoration. This Generative Facial Prior (GFP) is incorporated into the face restoration process via novel channel-split spatial feature transform layers, which allow our method to achieve a good balance of realness and fidelity. Thanks to the powerful generative facial prior and delicate designs, our GFP-GAN could jointly restore facial details and enhance colors with just a single forward pass, while GAN inversion methods require expensive image-specific optimization at inference. Extensive experiments show that our method achieves superior performance to prior art on both synthetic and real-world datasets.

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

### Installation

1. Clone repo

    ```bash
    git clone https://github.com/xinntao/GFPGAN.git
    ```

1. Install dependent packages

    ```bash
    cd GFPGAN
    pip install -r requirements.txt
    ```

## :zap: Quick Inference

> python inference_gfpgan_full.py --model_path experiments/pretrained_models/GFPGANv1.pth --test_path inputs

## :scroll: License and Acknowledgement

TODO

## :e-mail: Contact

If you have any question, please email `xintao.wang@outlook.com` or `xintaowang@tencent.com`.

# GFPGAN (CVPR 2021)

[English](README.md) **|** [简体中文](README_CN.md)

[[Paper]](https://arxiv.org/abs/2101.04061) **|** [[Project Page]](https://xinntao.github.io/projects/gfpgan)

GFPGAN is a blind face restoration algorithm towards real-world face images.

### GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior
[Xintao Wang](https://xinntao.github.io/), [Yu Li](https://yu-li.github.io/), [Honglun Zhang](https://scholar.google.com/citations?hl=en&user=KjQLROoAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en)
Applied Research Center (ARC), Tencent PCG

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

# Installation

We now provide a *clean* version of GFPGAN, which does not require customized CUDA extensions. See [here](README.md#installation) for this easier installation.<br>
If you want want to use the original model in our paper, please follow the instructions below.

1. Clone repo

    ```bash
    git clone https://github.com/xinntao/GFPGAN.git
    cd GFPGAN
    ```

1. Install dependent packages

    As StyleGAN2 uses customized PyTorch C++ extensions, you need to **compile them during installation** or **load them just-in-time(JIT)**.
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
    python setup.py develop

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
    python setup.py develop
    ```

## :zap: Quick Inference

Download pre-trained models: [GFPGANv1.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth)

```bash
wget https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth -P experiments/pretrained_models
```

- Option 1: Load extensions just-in-time(JIT)

    ```bash
    BASICSR_JIT=True python inference_gfpgan.py --input inputs/whole_imgs --output results --version 1

    # for aligned images
    BASICSR_JIT=True python inference_gfpgan.py --input inputs/whole_imgs --output results --version 1 --aligned
    ```

- Option 2: Have successfully compiled extensions during installation

    ```bash
    python inference_gfpgan.py --input inputs/whole_imgs --output results --version 1

    # for aligned images
    python inference_gfpgan.py --input inputs/whole_imgs --output results --version 1 --aligned
    ```

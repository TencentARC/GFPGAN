FROM nvidia/cuda:10.0-cudnn7-devel

ENV BASICSR_JIT=True

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    # python
    python3.8 python3-pip python3-setuptools python3-dev \
    # OpenCV deps
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx \
    # c++
    g++ \
    # others
    wget unzip

# Ninja
RUN wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip && \
    unzip ninja-linux.zip -d /usr/local/bin/ && \
    update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 


RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir basicsr

RUN pip3 install --no-cache-dir facexlib

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

RUN wget https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth \
        -P experiments/pretrained_models

RUN rm -rf /var/cache/apt/* /var/lib/apt/lists/* && \
    apt-get autoremove -y && apt-get clean

COPY . .

CMD ["python3", "inference_gfpgan_full.py", "--model_path", "experiments/pretrained_models/GFPGANv1.pth", \
     "--test_path", "inputs/whole_imgs"]

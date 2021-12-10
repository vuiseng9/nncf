FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-transport-https \
        git && \
rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        vim \
        wget \
        curl \
        zip \
        unzip \
        nano \
        openssh-server \
        openssh-client \
        sudo \
        graphviz libgraphviz-dev libgl1-mesa-glx \
    && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/miniconda/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh && \
    bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p /miniconda && \
    rm -f Miniconda3-py38_4.10.3-Linux-x86_64.sh && \
    conda --version && python --version 

# ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# ENV TORCH_VERSION=1.8.1
# ENV TORCHVISION_VERSION=0.9.1
# ENV MMCV_VERSION=1.3.0
# ENV CUDA_VERSION_CODE=102
# ENV MMDETECTION_DIR=ov-train-ext/external/mmdetection

# RUN git clone https://github.com/openvinotoolkit/training_extensions ov-train-ext && \
#     cd ov-train-ext && git submodule update --init external/mmdetection && \
#     pip install -e ote && \
#     pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} && \
#     pip uninstall -y mmcv && \
#     pip install cython && \
#     pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools" && \
#     TORCH_CUDA_ARCH_LIST="5.2 6.1 7.0" pip install --no-cache-dir mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA_VERSION_CODE}/torch${TORCH_VERSION}/index.html && \
#     pip install -e external/mmdetection && pip uninstall nncf

RUN git clone https://github.com/vuiseng9/nncf && \
    cd nncf && git checkout qaas-dec2021 && \
    python setup.py develop && \
    pip install -r examples/torch/requirements.txt && \
    pip install ipython jupyterlab pygraphviz more_itertools && \
    rm -rf ~/.cache/pip/* && \
    ninja_executable=$(python -c "import ninja; print(ninja.__path__[0])")/data/bin/ninja && \
    chmod +x $ninja_executable && \
    python -c "import nncf; print(\"NNCF {} is installed\".format(nncf.__version__))"

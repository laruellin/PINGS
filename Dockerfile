ARG PYTORCH="2.5.1"
ARG CUDA="11.8"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000

# Set NVIDIA environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV QT_X11_NO_MITSHM=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
        libglfw3-dev \
        libglib2.0-0 \
        libglm-dev \
        pkg-config \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY submodules/ ./submodules/
COPY requirements.txt .
RUN pip install --upgrade pip && \
    python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)" && \
    pip install -r requirements.txt --no-build-isolation

RUN if ! getent group "${GROUP_ID}" >/dev/null; then addgroup --gid "${GROUP_ID}" user; fi && \
    if ! id -u "${USER_ID}" >/dev/null 2>&1; then \
    adduser --disabled-password --gecos '' --uid "${USER_ID}" --gid "${GROUP_ID}" user; \
    fi
USER user

WORKDIR /packages/pings

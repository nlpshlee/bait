# BAIT: Bias Absorption via Internal-knowledge Textualization
Official implementation of "Catching the Bias": Suppressing confirmation bias in RAG by diverting parametric energy into generated decoy documents.

## Install Guide

Anaconda 가상 환경 생성
```bash
conda create -n bait python=3.11.7 -y
```

라이브러리 설치
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # CUDA 12.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 # CUDA 12.8
conda install -c nvidia cuda-toolkit # nvcc -V # 11.7 이상
pip install flash-attn==2.7.0.post2 --no-build-isolation
pip install -r requirements.txt
```
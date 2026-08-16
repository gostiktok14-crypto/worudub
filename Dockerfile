FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE_DIR=/runpod-volume/models \
    HF_HOME=/runpod-volume/models/huggingface \
    MODELSCOPE_CACHE=/runpod-volume/models/modelscope

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git git-lfs build-essential libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN printf "torch==2.3.1+cu121\ntorchaudio==2.3.1+cu121\n" > /app/torch-constraints.txt
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.3.1+cu121" "torchaudio==2.3.1+cu121"
RUN python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -c /app/torch-constraints.txt -r /app/requirements.txt
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-deps "torch==2.3.1+cu121" "torchaudio==2.3.1+cu121"
RUN python - <<'PY'
import torch
import torchaudio
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchaudio", torchaudio.__version__)
PY

RUN mkdir -p /app/vendor \
    && git clone https://github.com/biodatlab/thonburian-tts.git /app/vendor/thonburian-tts \
    && git -C /app/vendor/thonburian-tts checkout 032fe7e51674afe066a98e6d3cf47fc96d04b290 \
    && git clone https://github.com/FunAudioLLM/CosyVoice.git /app/vendor/CosyVoice \
    && git -C /app/vendor/CosyVoice checkout 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc \
    && git -C /app/vendor/CosyVoice submodule update --init --recursive

COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]

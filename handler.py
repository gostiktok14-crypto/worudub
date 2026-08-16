from __future__ import annotations

import base64
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import runpod
import soundfile as sf
import torch
from cached_path import cached_path
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/runpod-volume/models"))
HF_CACHE_DIR = Path(os.getenv("HF_HOME", str(CACHE_DIR / "huggingface")))
THONBURIAN_DIR = Path(os.getenv("THONBURIAN_SOURCE_DIR", ROOT / "vendor" / "thonburian-tts"))
COSYVOICE_DIR = Path(os.getenv("COSYVOICE_SOURCE_DIR", ROOT / "vendor" / "CosyVoice"))

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("MODELSCOPE_CACHE", str(CACHE_DIR / "modelscope"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_MODEL_LOCK = threading.Lock()
_JAITTS_MODEL: Any | None = None
_COSYVOICE_MODEL: Any | None = None


def _progress(job: dict[str, Any], message: str) -> None:
    try:
        runpod.serverless.progress_update(job, message)
    except Exception:
        pass


def _require_text(value: Any, field: str, max_length: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Missing required field: {field}")
    if len(text) > max_length:
        raise ValueError(f"{field} is too long (max {max_length} characters)")
    return text


def _decode_reference(data: dict[str, Any], work_dir: Path) -> Path:
    encoded = _require_text(data.get("ref_audio_base64"), "ref_audio_base64", 28_000_000)
    try:
        audio_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("ref_audio_base64 is not valid base64") from exc
    if not audio_bytes or len(audio_bytes) > 20 * 1024 * 1024:
        raise ValueError("Reference audio must be between 1 byte and 20 MB")
    suffix = Path(str(data.get("ref_audio_filename") or "reference.wav")).suffix.lower()
    if suffix not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
        suffix = ".wav"
    path = work_dir / f"reference{suffix}"
    path.write_bytes(audio_bytes)
    return path


def _normalize_reference(source: Path, work_dir: Path) -> Path:
    target = work_dir / "reference_24k.wav"
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-ac", "1", "-ar", "24000", str(target),
    ]
    import subprocess

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not target.exists():
        raise RuntimeError(f"Could not normalize reference audio: {completed.stderr[-600:]}")
    return target


def _load_jaitts() -> Any:
    global _JAITTS_MODEL
    if _JAITTS_MODEL is not None:
        return _JAITTS_MODEL
    with _MODEL_LOCK:
        if _JAITTS_MODEL is not None:
            return _JAITTS_MODEL
        if not torch.cuda.is_available():
            raise RuntimeError("JaiTTS worker requires a CUDA GPU")
        sys.path.insert(0, str(THONBURIAN_DIR))
        from flowtts.load_flowtts import FlowTTS

        model_path = Path(cached_path(
            os.getenv("JAITTS_CHECKPOINT", "hf://JTS-AI/JaiTTS-F5TTS/model.pt"),
            cache_dir=str(HF_CACHE_DIR),
        ))
        vocab_path = Path(cached_path(
            os.getenv("JAITTS_VOCAB", "hf://JTS-AI/JaiTTS-F5TTS/vocab.txt"),
            cache_dir=str(HF_CACHE_DIR),
        ))
        vocoder_dir = Path(snapshot_download(
            repo_id="charactr/vocos-mel-24khz",
            cache_dir=str(HF_CACHE_DIR),
        ))
        _JAITTS_MODEL = FlowTTS(
            device="cuda", model_type="F5", language="th", vocoder_name="vocos",
            checkpoint=str(model_path), vocab_file=str(vocab_path),
            local_path=str(vocoder_dir), ode_method="euler", use_ema=True,
            hf_cache_dir=str(HF_CACHE_DIR),
        )
        return _JAITTS_MODEL


def _load_cosyvoice() -> Any:
    global _COSYVOICE_MODEL
    if _COSYVOICE_MODEL is not None:
        return _COSYVOICE_MODEL
    with _MODEL_LOCK:
        if _COSYVOICE_MODEL is not None:
            return _COSYVOICE_MODEL
        if not torch.cuda.is_available():
            raise RuntimeError("CosyVoice worker requires a CUDA GPU")
        matcha_dir = COSYVOICE_DIR / "third_party" / "Matcha-TTS"
        sys.path.insert(0, str(COSYVOICE_DIR))
        if matcha_dir.exists():
            sys.path.insert(0, str(matcha_dir))
        from cosyvoice.cli.cosyvoice import AutoModel

        model_id = os.getenv("COSYVOICE_MODEL", "iic/Fun-CosyVoice3-0.5B")
        _COSYVOICE_MODEL = AutoModel(model_dir=model_id, fp16=True, load_vllm=False)
        return _COSYVOICE_MODEL


def _jaitts_generate(data: dict[str, Any], ref_audio: Path, output_path: Path) -> int:
    model = _load_jaitts()
    text = _require_text(data.get("text"), "text", 3000)
    ref_text = _require_text(data.get("ref_text"), "ref_text", 3000)
    options = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    speed = max(0.5, min(1.5, float(options.get("tts_speed", 1.0))))
    nfe_step = max(4, min(64, int(options.get("tts_nfe_step", 32))))
    cfg_strength = max(0.5, min(5.0, float(options.get("tts_cfg_strength", 2.0))))
    target_rms = max(0.03, min(0.3, float(options.get("target_rms", 0.1))))
    cross_fade = max(0.0, min(1.0, float(options.get("cross_fade_duration", 0.15))))
    if text[-1:] not in ".!?…。！？":
        text += "."
    spoken_chars = len(re.sub(r"[\s,.;:!?…。]+", "", text))
    fix_duration = float(sf.info(str(ref_audio)).duration) + max(1.4, spoken_chars * 0.082 / speed) + 0.5
    result = model.infer(
        ref_file=str(ref_audio), ref_text=ref_text, gen_text=text,
        show_info=lambda message: print(str(message), flush=True),
        target_rms=target_rms, cross_fade_duration=cross_fade,
        nfe_step=nfe_step, cfg_strength=cfg_strength, sway_sampling_coef=0.0,
        speed=speed, fix_duration=fix_duration, file_wave=str(output_path), seed=-1,
    )
    if (not output_path.exists() or output_path.stat().st_size < 1000) and isinstance(result, tuple) and len(result) >= 2:
        sf.write(str(output_path), result[0], int(result[1]))
    return 24000


def _cosyvoice_generate(data: dict[str, Any], ref_audio: Path, output_path: Path) -> int:
    model = _load_cosyvoice()
    text = _require_text(data.get("text"), "text", 3000)
    ref_text = str(data.get("ref_text") or "").strip()
    options = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    speed = max(0.5, min(2.0, float(options.get("tts_speed", 1.0))))
    instruction = str(options.get("speech_style_prompt") or "").strip()
    if ref_text and hasattr(model, "inference_zero_shot"):
        stream = model.inference_zero_shot(
            text, f"{instruction}{ref_text}", str(ref_audio), stream=False,
            speed=speed, text_frontend=True,
        )
    else:
        stream = model.inference_instruct2(
            text, instruction or "Speak naturally.", str(ref_audio), stream=False,
            speed=speed, text_frontend=True,
        )
    chunks = []
    for item in stream:
        speech = item["tts_speech"]
        chunks.append(speech.detach().cpu() if hasattr(speech, "detach") else speech)
    if not chunks:
        raise RuntimeError("CosyVoice returned no audio")
    audio = torch.cat(chunks, dim=-1) if len(chunks) > 1 else chunks[0]
    sample_rate = int(getattr(model, "sample_rate", 24000))
    sf.write(str(output_path), audio.squeeze().numpy(), sample_rate)
    return sample_rate


def handler(job: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    data = job.get("input")
    if not isinstance(data, dict):
        return {"status": "FAILED", "error": "input must be a JSON object"}
    if data.get("healthcheck") is True:
        return {
            "status": "READY", "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    if data.get("diagnostic") == "torch_audio":
        import torchaudio

        return {
            "status": "READY",
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torchaudio": torchaudio.__version__,
            "cuda": torch.cuda.is_available(),
        }
    work_dir = Path(tempfile.mkdtemp(prefix="runpod-tts-"))
    try:
        _progress(job, "Preparing reference audio")
        ref_audio = _normalize_reference(_decode_reference(data, work_dir), work_dir)
        output_path = work_dir / "output.wav"
        engine = str(data.get("engine") or "jaitts").strip().lower()
        _progress(job, f"Loading {engine} model")
        if engine in {"jaitts", "jai", "thai"}:
            sample_rate = _jaitts_generate(data, ref_audio, output_path)
            engine = "jaitts"
        elif engine in {"cosyvoice", "cosyvoice3", "cosy"}:
            sample_rate = _cosyvoice_generate(data, ref_audio, output_path)
            engine = "cosyvoice3"
        else:
            raise ValueError(f"Unsupported engine: {engine}")
        if not output_path.exists() or output_path.stat().st_size < 1000:
            raise RuntimeError("TTS engine did not create a valid audio file")
        _progress(job, "Encoding result")
        return {
            "status": "COMPLETED",
            "audio_base64": base64.b64encode(output_path.read_bytes()).decode("ascii"),
            "extension": "wav", "sample_rate": sample_rate, "engine": engine,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    except Exception as exc:
        print(f"Worker error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# RunPod's GitHub repository scanner looks for this registration call.
# Keep it at module scope so this image is recognized as a Queue worker.
runpod.serverless.start({"handler": handler})

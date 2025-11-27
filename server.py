# ---------------------------------------------------------
# MMAudio FastAPI Server (Stable Version with Debug Logging)
# ---------------------------------------------------------
import io
import os
import logging
import tempfile
from pathlib import Path

import torch
import torchaudio
import anyio

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

# MMAudio imports
from mmaudio.eval_utils import (
    ModelConfig,
    all_model_cfg,
    load_video,
    generate,
    make_video,
    setup_eval_logging
)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
log = logging.getLogger("mmaudio-fastapi")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="MMAudio API", version="1.0")

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
VARIANT = "large_44k_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

DEBUG_DIR = Path("/workspace/mmaudio-debug")
DEBUG_DIR.mkdir(exist_ok=True)

# Global model objects
model_cfg = None
seq_cfg = None
net = None
feature_utils = None


# ---------------------------------------------------------
# Startup — load model ONCE
# ---------------------------------------------------------
@app.on_event("startup")
def load_mmaudio():
    global model_cfg, seq_cfg, net, feature_utils

    setup_eval_logging()

    log.info(f"Loading MMAudio Variant = {VARIANT}")

    if VARIANT not in all_model_cfg:
        raise RuntimeError(f"Invalid model variant {VARIANT}")

    model_cfg = all_model_cfg[VARIANT]
    model_cfg.download_if_needed()
    seq_cfg = model_cfg.seq_cfg

    log.info(f"Using device={DEVICE} dtype={DTYPE}")

    net = get_my_mmaudio(model_cfg.model_name).to(DEVICE, DTYPE).eval()
    weights = torch.load(model_cfg.model_path, map_location=DEVICE, weights_only=True)
    net.load_weights(weights)
    log.info(f"Weights loaded: {model_cfg.model_path}")

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model_cfg.vae_path,
        synchformer_ckpt=model_cfg.synchformer_ckpt,
        enable_conditions=True,
        mode=model_cfg.mode,
        bigvgan_vocoder_ckpt=model_cfg.bigvgan_16k_path,
        need_vae_encoder=False
    ).to(DEVICE, DTYPE)
    feature_utils.eval()

    log.info("MMAudio model & utilities loaded successfully")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def run_blocking(func, *args, **kwargs):
    return anyio.to_thread.run_sync(lambda: func(*args, **kwargs))


def normalize_audio_tensor(audio):
    """
    Ensures audio is always a CPU torch tensor with shape (channels, samples)
    """
    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(audio)

    audio = audio.detach().cpu()

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    elif audio.dim() == 2:
        if audio.shape[0] > audio.shape[1] and audio.shape[1] <= 8:
            audio = audio.t()

    elif audio.dim() > 2:
        audio = audio.reshape(1, -1)

    if audio.dim() != 2:
        raise RuntimeError(f"Audio must be 2D after normalization. Got shape: {audio.shape}")

    return audio.float()


# ---------------------------------------------------------
# Endpoint: TEXT → AUDIO
# ---------------------------------------------------------
@app.post("/generate_audio")
async def generate_audio(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    duration: float = Form(8.0),
    cfg_strength: float = Form(4.5),
    num_steps: int = Form(25)
):
    try:
        def gen_fn():
            rng = torch.Generator(device=DEVICE)
            rng.manual_seed(42)
            fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

            seq_cfg.duration = duration
            net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

            audios = generate(
                clip_frames=None,
                sync_frames=None,
                text=[prompt],
                negative_text=[negative_prompt],
                feature_utils=feature_utils,
                net=net,
                fm=fm,
                rng=rng,
                cfg_strength=cfg_strength
            )
            return audios.float().cpu()[0]

        audio = await run_blocking(gen_fn)
        audio = normalize_audio_tensor(audio)

        output_path = DEBUG_DIR / f"audio_{abs(hash(prompt))}.flac"
        torchaudio.save(str(output_path), audio, seq_cfg.sampling_rate, format="FLAC")

        log.info(f"[AUDIO] Saved → {output_path} size={output_path.stat().st_size}")

        return FileResponse(str(output_path), media_type="audio/flac")

    except Exception as e:
        log.exception("Error in /generate_audio")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# Endpoint: VIDEO → VIDEO with generated audio
# ---------------------------------------------------------
@app.post("/generate_video")
async def generate_video(
    video: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    duration: float = Form(8.0),
    cfg_strength: float = Form(4.5),
    num_steps: int = Form(25)
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename).suffix) as f:
            uploaded_video_path = Path(f.name)
            f.write(await video.read())

        log.info(f"[DEBUG] Uploaded video → {uploaded_video_path} "
                 f"size={uploaded_video_path.stat().st_size}")

        def gen_video_fn():
            rng = torch.Generator(device=DEVICE)
            rng.manual_seed(42)
            fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

            video_info = load_video(uploaded_video_path, duration)
            real_dur = video_info.duration_sec
            log.info(f"[DEBUG] Video duration = {real_dur}")

            clip_frames = video_info.clip_frames.unsqueeze(0)
            sync_frames = video_info.sync_frames.unsqueeze(0)

            seq_cfg.duration = real_dur
            net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

            audios = generate(
                clip_frames,
                sync_frames,
                [prompt],
                negative_text=[negative_prompt],
                feature_utils=feature_utils,
                net=net,
                fm=fm,
                rng=rng,
                cfg_strength=cfg_strength
            )

            audio = normalize_audio_tensor(audios.float().cpu()[0])

            audio_path = DEBUG_DIR / f"video_audio_{uploaded_video_path.stem}.flac"
            torchaudio.save(str(audio_path), audio, seq_cfg.sampling_rate, format="FLAC")

            log.info(f"[VIDEO_A] Audio saved → {audio_path}")

            final_path = DEBUG_DIR / f"final_{uploaded_video_path.stem}.mp4"
            make_video(video_info, final_path, audio, sampling_rate=seq_cfg.sampling_rate)

            log.info(f"[VIDEO] Final video saved → {final_path}")
            return final_path

        final_video_path = await run_blocking(gen_video_fn)

        return FileResponse(
            str(final_video_path),
            media_type="video/mp4",
            filename=final_video_path.name
        )

    except Exception as e:
        log.exception("Error in /generate_video")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"status": "MMAudio server running", "variant": VARIANT}

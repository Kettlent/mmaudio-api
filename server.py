# server.py
import logging
import io
import tempfile
import os
from pathlib import Path

import torch
import torchaudio
import anyio

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse

# MMAudio imports
from mmaudio.eval_utils import (
    ModelConfig,
    all_model_cfg,
    generate,
    load_video,
    make_video,
    setup_eval_logging
)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils


# -------------------------------------------------------------
# FASTAPI INIT
# -------------------------------------------------------------
app = FastAPI(title="MMAudio FastAPI")
log = logging.getLogger("mmaudio-fastapi")
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------------
# GLOBAL CONFIG
# -------------------------------------------------------------
VARIANT = "large_44k_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

model_cfg = None
seq_cfg = None
net = None
feature_utils = None


# -------------------------------------------------------------
# UTIL
# -------------------------------------------------------------
def run_blocking(fn, *args, **kwargs):
    """Run blocking function in a worker thread."""
    return anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))


def safe_tensor(x):
    """Clone, detach, contiguous, move to device and dtype."""
    if x is None:
        return None
    return (
        x.clone()
         .detach()
         .contiguous()
         .to(device=DEVICE, dtype=DTYPE)
    )


# -------------------------------------------------------------
# STARTUP — Load Model Once
# -------------------------------------------------------------
@app.on_event("startup")
def load_model():
    global model_cfg, seq_cfg, net, feature_utils

    setup_eval_logging()
    log.info(f"[INIT] Loading MMAudio variant: {VARIANT}")

    model_cfg = all_model_cfg[VARIANT]
    model_cfg.download_if_needed()
    seq_cfg = model_cfg.seq_cfg

    log.info(f"[INIT] Device: {DEVICE}, dtype: {DTYPE}")

    # Load main model
    net = get_my_mmaudio(model_cfg.model_name).to(DEVICE, DTYPE).eval()
    weights = torch.load(model_cfg.model_path, map_location=DEVICE, weights_only=True)
    net.load_weights(weights)
    log.info("[INIT] Loaded model weights")

    # Load feature utils
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model_cfg.vae_path,
        synchformer_ckpt=model_cfg.synchformer_ckpt,
        enable_conditions=True,
        mode=model_cfg.mode,
        bigvgan_vocoder_ckpt=model_cfg.bigvgan_16k_path,
        need_vae_encoder=False
    ).to(DEVICE, DTYPE).eval()

    log.info("[INIT] Feature utils ready")


# -------------------------------------------------------------
# GENERATE AUDIO ONLY
# -------------------------------------------------------------
@app.post("/generate_audio")
async def generate_audio(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    duration: float = Form(8.0),
    cfg_strength: float = Form(4.5),
    num_steps: int = Form(25),
):
    try:
        rng = torch.Generator(device=DEVICE).manual_seed(42)
        fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

        seq_cfg.duration = duration
        net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

        def gen_fn():
            with torch.no_grad():
                audio = generate(
                    None,
                    None,
                    [prompt],
                    negative_text=[negative_prompt],
                    feature_utils=feature_utils,
                    net=net,
                    fm=fm,
                    rng=rng,
                    cfg_strength=cfg_strength
                ).float().cpu()[0]
            return audio

        audio = await run_blocking(gen_fn)

        # Save to temp flac
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".flac").name
        torchaudio.save(temp_path, audio.unsqueeze(0), seq_cfg.sampling_rate, format="FLAC")

        return FileResponse(
            temp_path,
            media_type="audio/flac",
            filename="mmaudio_output.flac",
            background=None
        )

    except Exception as e:
        log.exception("Error in generate_audio")
        raise HTTPException(500, str(e))


# -------------------------------------------------------------
# GENERATE VIDEO (video → new video with generated audio)
# -------------------------------------------------------------
@app.post("/generate_video")
async def generate_video(
    video: UploadFile = File(...),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    duration: float = Form(8.0),
    cfg_strength: float = Form(4.5),
    num_steps: int = Form(25),
):
    DEBUG_DIR = Path("/workspace/mmaudio-debug")
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # -------------------------------------------------
        # Save upload
        # -------------------------------------------------
        uploaded_path = DEBUG_DIR / f"uploaded_{video.filename}"
        with open(uploaded_path, "wb") as f:
            f.write(await video.read())

        log.info(f"[DEBUG] Saved upload → {uploaded_path} ({uploaded_path.stat().st_size} bytes)")

        if uploaded_path.stat().st_size < 2000:
            raise HTTPException(400, "Uploaded video too small or corrupted")

        # -------------------------------------------------
        # Heavy generation in a worker thread
        # -------------------------------------------------
        def gen_video_fn():

            rng = torch.Generator(device=DEVICE).manual_seed(42)
            fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

            # Load video
            video_info = load_video(uploaded_path, duration)

            clip = video_info.clip_frames
            sync = video_info.sync_frames

            log.info(f"[DEBUG] clip shape = {None if clip is None else clip.shape}")
            log.info(f"[DEBUG] sync shape = {None if sync is None else sync.shape}")

            # Prepare frames
            clip_frames = safe_tensor(clip.unsqueeze(0)) if clip is not None else None
            sync_frames = safe_tensor(sync.unsqueeze(0)) if sync is not None else None

            # Update sequence lengths
            seq_cfg.duration = video_info.duration_sec
            net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

            with torch.no_grad():
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
                audio = audios.float().cpu()[0]

            # Save audio debug
            audio_path = DEBUG_DIR / "generated_audio.flac"
            torchaudio.save(audio_path, audio.unsqueeze(0), seq_cfg.sampling_rate, format="FLAC")
            log.info(f"[DEBUG] Saved audio → {audio_path} ({audio_path.stat().st_size} bytes)")

            # Save final video
            output_path = DEBUG_DIR / "final_output.mp4"
            make_video(video_info, output_path, audio, sampling_rate=seq_cfg.sampling_rate)
            log.info(f"[DEBUG] Saved final video → {output_path} ({output_path.stat().st_size} bytes)")

            return output_path

        # Run generation
        final_path = await run_blocking(gen_video_fn)

        if final_path is None or not final_path.exists():
            raise HTTPException(500, "Video generation failed")

        # -------------------------------------------------
        # Return completed MP4
        # -------------------------------------------------
        return FileResponse(
            str(final_path),
            media_type="video/mp4",
            filename="mmaudio_generated.mp4"
        )

    except Exception as e:
        log.exception("Error in generate_video")
        raise HTTPException(500, str(e))


@app.get("/")
def root():
    return {"status": "ok", "variant": VARIANT}

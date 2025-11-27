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
from fastapi.responses import StreamingResponse

# MMAudio imports (same as demo.py)
from mmaudio.eval_utils import (
    ModelConfig, all_model_cfg, generate,
    load_video, make_video, setup_eval_logging
)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils


# ----------------------------------------------------
# Basic Setup
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mmaudio-fastapi")

app = FastAPI(title="MMAudio FastAPI")

VARIANT = "large_44k_v2"    # change if needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


# ----------------------------------------------------
# Global model objects (filled at startup)
# ----------------------------------------------------
model_cfg: ModelConfig | None = None
seq_cfg = None
net: MMAudio | None = None
feature_utils: FeaturesUtils | None = None


# ----------------------------------------------------
# Load Model Once on Startup
# ----------------------------------------------------
@app.on_event("startup")
def load_model():
    global model_cfg, seq_cfg, net, feature_utils

    setup_eval_logging()

    log.info(f"Loading MMAudio variant: {VARIANT}")
    if VARIANT not in all_model_cfg:
        raise RuntimeError(f"Unknown model variant: {VARIANT}")

    model_cfg = all_model_cfg[VARIANT]
    model_cfg.download_if_needed()
    seq_cfg = model_cfg.seq_cfg

    log.info(f"Device = {DEVICE}, dtype = {DTYPE}")

    # Load network
    net = get_my_mmaudio(model_cfg.model_name).to(DEVICE, DTYPE).eval()
    weights = torch.load(model_cfg.model_path, map_location=DEVICE, weights_only=True)
    net.load_weights(weights)
    log.info(f"Model weights loaded: {model_cfg.model_path}")

    # Load feature utils (VAE, synchformer, vocoder)
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model_cfg.vae_path,
        synchformer_ckpt=model_cfg.synchformer_ckpt,
        enable_conditions=True,
        mode=model_cfg.mode,
        bigvgan_vocoder_ckpt=model_cfg.bigvgan_16k_path,
        need_vae_encoder=False,
    ).to(DEVICE, DTYPE).eval()

    log.info("Feature utils loaded successfully")
    log.info("MMAudio is ready!")


# ----------------------------------------------------
# Run blocking function safely in thread
# ----------------------------------------------------
def run_blocking(fn, *args, **kwargs):
    return anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))


# ----------------------------------------------------
# /generate_audio — text → audio
# ----------------------------------------------------
@app.post("/generate_audio")
async def generate_audio(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    duration: float = Form(8.0),
    cfg_strength: float = Form(4.5),
    num_steps: int = Form(25)
):
    """
    Generate FLAC audio from a text prompt using MMAudio.
    """
    try:
        # Build FlowMatching
        rng = torch.Generator(device=DEVICE)
        rng.manual_seed(42)
        fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

        # Update sequence length for requested duration
        seq_cfg.duration = duration
        net.update_seq_lengths(
            seq_cfg.latent_seq_len,
            seq_cfg.clip_seq_len,
            seq_cfg.sync_seq_len
        )

        # Blocking generation (run in thread)
        def gen_fn():
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

        # Convert audio → FLAC in memory
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.unsqueeze(0), seq_cfg.sampling_rate, format="FLAC")
        buffer.seek(0)

        filename = f"mmaudio_{abs(hash(prompt))}.flac"

        return StreamingResponse(
            buffer,
            media_type="audio/flac",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        log.exception("Error in /generate_audio")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------
# /generate_video — video + text → new video with generated audio
# ----------------------------------------------------
@app.post("/generate_video")
async def generate_video(
    video: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    duration: float = Form(8.0),
    cfg_strength: float = Form(4.5),
    num_steps: int = Form(25),
):
    """
    Accepts an uploaded video file and returns an MP4 video where
    MMAudio-generated audio replaces the original track.
    """
    uploaded_path = None

    try:
        # Save uploaded video to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename).suffix) as vf:
            uploaded_path = Path(vf.name)
            vf.write(await video.read())

        log.info(f"Uploaded video stored at: {uploaded_path}")

        def gen_video_fn():
            rng = torch.Generator(device=DEVICE)
            rng.manual_seed(42)
            fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

            # Load frames from video
            video_info = load_video(uploaded_path, duration)
            clip_frames = (
                video_info.clip_frames.unsqueeze(0) if video_info.clip_frames is not None else None
            )
            sync_frames = (
                video_info.sync_frames.unsqueeze(0) if video_info.sync_frames is not None else None
            )

            # Adjust model for video duration
            seq_cfg.duration = video_info.duration_sec
            net.update_seq_lengths(
                seq_cfg.latent_seq_len,
                seq_cfg.clip_seq_len,
                seq_cfg.sync_seq_len
            )

            # Generate audio conditioned on video frames
            audios = generate(
                clip_frames,
                sync_frames,
                [prompt],
                negative_text=[negative_prompt],
                feature_utils=feature_utils,
                net=net,
                fm=fm,
                rng=rng,
                cfg_strength=cfg_strength,
            )
            audio = audios.float().cpu()[0]

            # Output video path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as out_video:
                saved_video_path = Path(out_video.name)

            make_video(video_info, saved_video_path, audio, sampling_rate=seq_cfg.sampling_rate)

            return saved_video_path

        # Run generation in worker thread
        new_video_path = await run_blocking(gen_video_fn)

        # Stream the MP4
        def streamer(path):
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 64)
                    if not chunk:
                        break
                    yield chunk
            # cleanup
            try:
                os.remove(path)
                os.remove(uploaded_path)
            except:
                pass

        return StreamingResponse(
            streamer(new_video_path),
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename=mmaudio_generated.mp4"
            }
        )

    except Exception as e:
        log.exception("Error in /generate_video")
        try:
            if uploaded_path and uploaded_path.exists():
                os.remove(uploaded_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------
# Root endpoint
# ----------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "variant": VARIANT, "device": DEVICE}

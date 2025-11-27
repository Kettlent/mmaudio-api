# server.py
import logging
import io
import tempfile
import os
from pathlib import Path
from argparse import Namespace

import torch
import torchaudio
import anyio

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

# MMAudio imports (same as demo.py)
from mmaudio.eval_utils import ModelConfig, all_model_cfg, generate, load_video, make_video, setup_eval_logging
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

log = logging.getLogger("mmaudio-fastapi")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="MMAudio FastAPI")

# ---------- CONFIG ----------
# Default variant (same choices as demo.py)
VARIANT = "large_44k_v2"   # change if you need other variant
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use bfloat16 on CUDA for lower memory if supported, otherwise float32
DTYPE = torch.bfloat16 if (DEVICE == "cuda") else torch.float32

# Global model holders (filled in startup)
model_cfg = None
seq_cfg = None
net = None
feature_utils = None

# ---------- STARTUP: load model once ----------
@app.on_event("startup")
def load_model():
    global model_cfg, seq_cfg, net, feature_utils

    setup_eval_logging()  # same as demo to setup logging formats if present

    log.info(f"Loading MMAudio variant: {VARIANT}")
    if VARIANT not in all_model_cfg:
        raise RuntimeError(f"Unknown model variant: {VARIANT}")

    model_cfg: ModelConfig = all_model_cfg[VARIANT]
    model_cfg.download_if_needed()
    seq_cfg = model_cfg.seq_cfg

    log.info(f"Device: {DEVICE}, dtype: {DTYPE}")

    # instantiate network and load weights
    net: MMAudio = get_my_mmaudio(model_cfg.model_name).to(DEVICE, DTYPE).eval()
    weights = torch.load(model_cfg.model_path, map_location=DEVICE, weights_only=True)
    net.load_weights(weights)
    log.info(f"Loaded weights from {model_cfg.model_path}")

    # feature utils (VAE, synchformer, vocoder)
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model_cfg.vae_path,
        synchformer_ckpt=model_cfg.synchformer_ckpt,
        enable_conditions=True,
        mode=model_cfg.mode,
        bigvgan_vocoder_ckpt=model_cfg.bigvgan_16k_path,
        need_vae_encoder=False
    ).to(DEVICE, DTYPE).eval()

    log.info("Feature utils and model loaded successfully")
    # optionally warm-up small step here if you want (skip to save time)

# ---------- Utility to run blocking generation off the event loop ----------
def run_blocking(fn, *args, **kwargs):
    return anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))

# ---------- /generate_audio endpoint ----------
@app.post("/generate_audio")
async def generate_audio(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    duration: float = Form(8.0),
    cfg_strength: float = Form(4.5),
    num_steps: int = Form(25),
):
    """
    Generate audio (FLAC) from text prompt.
    """
    try:
        # Prepare RNG and FlowMatching
        rng = torch.Generator(device=DEVICE)
        rng.manual_seed(42)
        fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

        # update seq lengths based on requested duration
        seq_cfg.duration = duration
        net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

        # run generation in thread (blocking heavy work)
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
            audio = audios.float().cpu()[0]  # (samples,)
            return audio

        audio = await run_blocking(gen_fn)

        # Write FLAC into memory buffer
        buffer = io.BytesIO()
        # torchaudio.save expects (channels, samples), so add batch dim
        torchaudio.save(buffer, audio.unsqueeze(0), seq_cfg.sampling_rate, format="FLAC")
        buffer.seek(0)

        filename = f"mmaudio_{abs(hash(prompt)) % 10_000}.flac"
        return StreamingResponse(buffer, media_type="audio/flac", headers={
            "Content-Disposition": f"attachment; filename={filename}"
        })

    except Exception as e:
        log.exception("Error in generate_audio")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- /generate_video endpoint ----------
@app.post("/generate_video")
async def generate_video(
    video: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    duration: float = Form(8.0),
    cfg_strength: float = Form(4.5),
    num_steps: int = Form(25),
    skip_video_composite: bool = Form(False)
):
    """
    Accepts an uploaded video file and returns a new MP4 where MMAudio generated audio
    is composited with the original video (using make_video).
    """
    # save upload to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename).suffix) as vf:
            uploaded_video_path = Path(vf.name)
            vf.write(await video.read())
        log.info(f"Saved uploaded video to {uploaded_video_path}")

        # load video and prepare frames (blocking part)
        def gen_video_fn():
            # set device RNG + fm per request
            rng = torch.Generator(device=DEVICE)
            rng.manual_seed(42)
            fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

            # load video info (this also sets clip_frames, sync_frames, duration)
            video_info = load_video(uploaded_video_path, duration)
            clip_frames = video_info.clip_frames.unsqueeze(0) if video_info.clip_frames is not None else None
            sync_frames = video_info.sync_frames.unsqueeze(0) if video_info.sync_frames is not None else None

            # adjust sequence config duration to match video
            seq_cfg.duration = video_info.duration_sec
            net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

            # generate audio conditioned on clip_frames/sync_frames
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
            audio = audios.float().cpu()[0]  # (samples, )

            # write audio and composite video to a temp mp4 path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as outf:
                video_save_path = Path(outf.name)
            # make_video will read video_info and write composite video to video_save_path
            make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
            return video_save_path

        video_save_path = await run_blocking(gen_video_fn)

        # stream the created mp4 back to caller
        def iterfile(path: Path):
            with open(path, "rb") as f:
                chunk = f.read(1024 * 32)
                while chunk:
                    yield chunk
                    chunk = f.read(1024 * 32)
            # cleanup temp files
            try:
                os.remove(str(path))
                os.remove(str(uploaded_video_path))
            except Exception:
                pass

        return StreamingResponse(iterfile(video_save_path), media_type="video/mp4",
                                 headers={"Content-Disposition": f"attachment; filename={video_save_path.name}"})

    except Exception as e:
        log.exception("Error in generate_video")
        # clean up uploaded file if exists
        try:
            if uploaded_video_path and uploaded_video_path.exists():
                os.remove(str(uploaded_video_path))
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"status": "mmaudio server running", "variant": VARIANT}

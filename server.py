# ---------------------------------------------------------
# MMAudio FastAPI Server (Stable Version with Debug Logging)
# ---------------------------------------------------------

#ssh -i /Users/scallercell_2/Desktop/cosyvoice root@69.30.85.218 -p 22169
#python3 server.py --port 8888 --model_dir ../../../pretrained_models/CosyVoice2-0.5B
#podid - https://hmwecuisc92c1a-8888.proxy.runpod.net

#ssh -i /Users/scallercell_2/Desktop/cosyvoice root@74.2.96.22 -p 15815 

#ssh -i /Users/scallercell_2/Desktop/cosyvoice root@69.30.85.167 -p 22098 


# cd /workspace
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
# bash miniconda.sh -b -p /workspace/miniconda
#source /workspace/miniconda/etc/profile.d/conda.sh


# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# conda create -y -n cosyvoice python=3.10
# conda activate cosyvoice

#echo "source /workspace/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc

# pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com


#chmod +x /workspace/start.sh
# bash /workspace/start.sh
# tail -50 /workspace/cosyvoice.log


# server.py
import logging
import io
import os
import tempfile
from pathlib import Path
import uuid
import asyncio

import torch
import torchaudio
import anyio

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

# MMAudio imports
from mmaudio.eval_utils import (
    ModelConfig, all_model_cfg, generate, load_video, make_video, setup_eval_logging
)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils


# ---------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------
log = logging.getLogger("mmaudio-fastapi")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="MMAudio FastAPI (SSE Enabled)")

# ---------------------------------------------------------
# GLOBALS
# ---------------------------------------------------------

VARIANT = "large_44k_v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

model_cfg = None
seq_cfg = None
net = None
feature_utils = None

# Job tracking
job_status = {}   # job_id → dict(status, progress, result, error)
job_locks = {}    # job_id → asyncio.Lock()

DEBUG_DIR = Path("/workspace/mmaudio-debug")
DEBUG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

async def run_blocking(fn, *args, **kwargs):
    """Runs CPU/GPU heavy work in thread-safe worker."""
    return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))


def update_progress(job_id: str, progress: int):
    if job_id in job_status:
        job_status[job_id]["progress"] = progress
        log.info(f"[JOB {job_id}] Progress = {progress}%")


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
@app.on_event("startup")
def load_model():
    global model_cfg, seq_cfg, net, feature_utils

    setup_eval_logging()
    log.info(f"Loading MMAudio model variant: {VARIANT}")

    model_cfg = all_model_cfg[VARIANT]
    model_cfg.download_if_needed()
    seq_cfg = model_cfg.seq_cfg

    log.info(f"Device: {DEVICE}, dtype: {DTYPE}")

    # Load network
    net = get_my_mmaudio(model_cfg.model_name).to(DEVICE, DTYPE).eval()
    weights = torch.load(model_cfg.model_path, map_location=DEVICE, weights_only=True)
    net.load_weights(weights)

    # Feature utils
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model_cfg.vae_path,
        synchformer_ckpt=model_cfg.synchformer_ckpt,
        enable_conditions=True,
        mode=model_cfg.mode,
        bigvgan_vocoder_ckpt=model_cfg.bigvgan_16k_path,
        need_vae_encoder=False
    ).to(DEVICE, DTYPE).eval()

    log.info("MMAudio model & utilities loaded successfully.")


# ---------------------------------------------------------
# GENERATE VIDEO (Runs as background job)
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
    job_id = str(uuid.uuid4())

    # Create job entry
    job_status[job_id] = {
        "status": "queued",
        "progress": 0,
        "result": None,
        "error": None
    }
    job_locks[job_id] = asyncio.Lock()

    # Save uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        uploaded_video_path = Path(f.name)
        f.write(await video.read())

    log.info(f"[JOB {job_id}] Saved uploaded video → {uploaded_video_path}")

    # Background task --------------------------------------
    async def job_thread():
        async with job_locks[job_id]:
            try:
                job_status[job_id]["status"] = "processing"
                update_progress(job_id, 5)

                # Load video
                video_info = await run_blocking(load_video, uploaded_video_path, duration)
                update_progress(job_id, 20)

                # Prepare RNG/FlowMatching
                fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)
                rng = torch.Generator(device=DEVICE)
                rng.manual_seed(42)

                # Update seq config to match actual video duration
                seq_cfg.duration = video_info.duration_sec
                net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

                # Generate audio
                def generate_audio_block():
                    return generate(
                        video_info.clip_frames.unsqueeze(0),
                        video_info.sync_frames.unsqueeze(0),
                        [prompt],
                        negative_text=[negative_prompt],
                        feature_utils=feature_utils,
                        net=net,
                        fm=fm,
                        rng=rng,
                        cfg_strength=cfg_strength
                    ).float().cpu()[0]

                audio = await run_blocking(generate_audio_block)
                update_progress(job_id, 70)

                # Save audio
                audio_path = DEBUG_DIR / f"{job_id}.flac"
                torchaudio.save(audio_path, audio.unsqueeze(0), seq_cfg.sampling_rate, format="FLAC")
                update_progress(job_id, 85)

                # Generate final MP4
                final_video_path = DEBUG_DIR / f"{job_id}.mp4"
                await run_blocking(make_video, video_info, final_video_path, audio, seq_cfg.sampling_rate)

                update_progress(job_id, 100)

                job_status[job_id]["status"] = "done"
                job_status[job_id]["result"] = str(final_video_path)

                log.info(f"[JOB {job_id}] Completed → {final_video_path}")

            except Exception as e:
                job_status[job_id]["status"] = "error"
                job_status[job_id]["error"] = str(e)
                log.error(f"[JOB {job_id}] ERROR → {e}")

    # Start background job
    asyncio.create_task(job_thread())

    # Return job_id immediately
    return {"job_id": job_id, "status": "queued"}


# ---------------------------------------------------------
# SSE PROGRESS STREAM
# ---------------------------------------------------------
@app.get("/progress/{job_id}")
async def progress_stream(job_id: str):

    async def event_gen():
        last_progress = -1

        while True:
            job = job_status.get(job_id)
            if not job:
                yield "event: error\ndata: Job not found\n\n"
                break

            status = job["status"]
            progress = job["progress"]

            # Send progress only if new
            if progress != last_progress:
                last_progress = progress
                yield f"event: progress\ndata: {progress}\n\n"

            if status == "done":
                result = job["result"]
                yield f"event: done\ndata: {result}\n\n"
                break

            if status == "error":
                yield f"event: error\ndata: {job['error']}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------------------------------------------
# ROOT
# ---------------------------------------------------------
@app.get("/")
def root():
    return {"status": "mmaudio server running", "variant": VARIANT}
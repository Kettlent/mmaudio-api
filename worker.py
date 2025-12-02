import os
import time
import json
import uuid
import torch
import logging
from pathlib import Path

from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate,
                                load_video, make_video)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mmaudio-worker")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
VARIANT = "large_44k_v2"

JOB_DIR = Path("./jobs")
JOB_DIR.mkdir(exist_ok=True)

def load_model():
    log.info("Loading MMAudio model...")
    model_cfg: ModelConfig = all_model_cfg[VARIANT]
    model_cfg.download_if_needed()

    net: MMAudio = get_my_mmaudio(model_cfg.model_name).to(DEVICE, DTYPE).eval()
    net.load_weights(torch.load(model_cfg.model_path, map_location=DEVICE, weights_only=True))

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model_cfg.vae_path,
        synchformer_ckpt=model_cfg.synchformer_ckpt,
        enable_conditions=True,
        mode=model_cfg.mode,
        bigvgan_vocoder_ckpt=model_cfg.bigvgan_16k_path,
        need_vae_encoder=False
    ).to(DEVICE, DTYPE).eval()

    return model_cfg, net, feature_utils

model_cfg, net, feature_utils = load_model()


def update_progress(job_id, pct):
    (JOB_DIR / f"{job_id}_progress.txt").write_text(str(pct))


def update_status(job_id, msg):
    (JOB_DIR / f"{job_id}_status.txt").write_text(msg)


def worker_loop():
    log.info("Worker started. Waiting for jobs...")

    while True:
        queue_path = JOB_DIR / "queue.json"

        if not queue_path.exists():
            time.sleep(0.5)
            continue

        try:
            queue = json.loads(queue_path.read_text())
        except:
            time.sleep(0.5)
            continue

        if len(queue) == 0:
            time.sleep(0.3)
            continue

        job = queue.pop(0)
        queue_path.write_text(json.dumps(queue))

        job_id = job["job_id"]
        prompt = job["prompt"]
        input_path = Path(job["input_path"])
        duration = float(job["duration"])
        cfg_strength = float(job["cfg_strength"])
        num_steps = int(job["num_steps"])

        update_status(job_id, "processing")

        try:
            # Step 1: load video
            update_progress(job_id, 5)
            video_info = load_video(input_path, duration)
            clip_frames = video_info.clip_frames.unsqueeze(0) if video_info.clip_frames is not None else None
            sync_frames = video_info.sync_frames.unsqueeze(0) if video_info.sync_frames is not None else None

            # Step 2: setup FM + RNG
            update_progress(job_id, 15)
            rng = torch.Generator(device=DEVICE)
            rng.manual_seed(42)
            fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

            # Step 3: match seq length
            update_progress(job_id, 30)
            model_cfg.seq_cfg.duration = video_info.duration_sec
            net.update_seq_lengths(model_cfg.seq_cfg.latent_seq_len,
                                   model_cfg.seq_cfg.clip_seq_len,
                                   model_cfg.seq_cfg.sync_seq_len)

            # Step 4: generate audio
            update_progress(job_id, 60)
            with torch.inference_mode():
                audios = generate(
                    clip_frames,
                    sync_frames,
                    [prompt],
                    negative_text=[""],
                    feature_utils=feature_utils,
                    net=net,
                    fm=fm,
                    rng=rng,
                    cfg_strength=cfg_strength
                )

            audio = audios.float().cpu()[0]

            # Step 5: write final mp4
            update_progress(job_id, 80)
            output_file = JOB_DIR / f"{job_id}_output.mp4"
            make_video(video_info, output_file, audio, sampling_rate=model_cfg.seq_cfg.sampling_rate)

            update_progress(job_id, 100)
            update_status(job_id, "done")

        except Exception as e:
            update_status(job_id, f"error: {e}")
            log.exception(f"[JOB {job_id}] ERROR: {e}")

        time.sleep(0.1)


if __name__ == "__main__":
    worker_loop()

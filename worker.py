import json
import time
import os
from pathlib import Path
import logging
import torch
import torchaudio

from mmaudio.eval_utils import (
    ModelConfig, all_model_cfg, generate, load_video, make_video
)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mmaudio-worker")

BASE_DIR = Path("/workspace/mmaudio-api")
JOB_DIR = BASE_DIR / "jobs"
QUEUE_FILE = BASE_DIR / "queue.json"

JOB_DIR.mkdir(exist_ok=True)

VARIANT = "large_44k_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def load_model():
    log.info("Loading MMAudio model...")

    model_cfg: ModelConfig = all_model_cfg[VARIANT]
    model_cfg.download_if_needed()

    net: MMAudio = get_my_mmaudio(model_cfg.model_name).to(DEVICE, DTYPE).eval()
    weights = torch.load(model_cfg.model_path, map_location=DEVICE, weights_only=True)
    net.load_weights(weights)

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model_cfg.vae_path,
        synchformer_ckpt=model_cfg.synchformer_ckpt,
        enable_conditions=True,
        mode=model_cfg.mode,
        bigvgan_vocoder_ckpt=model_cfg.bigvgan_16k_path,
        need_vae_encoder=False
    ).to(DEVICE, DTYPE).eval()

    seq_cfg = model_cfg.seq_cfg

    log.info("Model loaded successfully.")
    return net, feature_utils, seq_cfg, model_cfg


def set_progress(job_id: str, value: int | float):
    (JOB_DIR / job_id / "progress.txt").write_text(str(value))


def set_status(job_id: str, value: str):
    (JOB_DIR / job_id / "status.txt").write_text(value)


def process_job(job_id: str, job_data: dict, net: MMAudio, feature_utils: FeaturesUtils, seq_cfg):
    job_folder = JOB_DIR / job_id
    job_folder.mkdir(exist_ok=True)

    input_video = job_data["input_video"]  # full path string
    prompt = job_data["prompt"]
    negative_prompt = job_data["negative_prompt"]
    duration = float(job_data["duration"])
    cfg_strength = float(job_data["cfg_strength"])
    num_steps = int(job_data["num_steps"])

    # We already saved the uploaded file into this path in server.py
    local_video_path = Path(input_video)
    log.info(f"[JOB {job_id}] Using uploaded video at {local_video_path}")

    try:
        set_status(job_id, "running")
        set_progress(job_id, 5)

        # 1) Load video and extract frames
        video_info = load_video(local_video_path, duration)

        clip_frames = (
            video_info.clip_frames.unsqueeze(0)
            if video_info.clip_frames is not None
            else None
        )
        sync_frames = (
            video_info.sync_frames.unsqueeze(0)
            if video_info.sync_frames is not None
            else None
        )

        # 2) Match sequence config to video duration
        seq_cfg.duration = video_info.duration_sec
        net.update_seq_lengths(
            seq_cfg.latent_seq_len,
            seq_cfg.clip_seq_len,
            seq_cfg.sync_seq_len
        )

        rng = torch.Generator(device=DEVICE)
        rng.manual_seed(42)
        fm = FlowMatching(
            min_sigma=0,
            inference_mode="euler",
            num_steps=num_steps
        )

        set_progress(job_id, 30)

        # 3) Generate audio
        with torch.inference_mode():
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

        audio = audios.float().cpu()[0]  # (samples,)

        audio_path = job_folder / "audio.flac"
        torchaudio.save(
            str(audio_path),
            audio.unsqueeze(0),  # (1, samples)
            seq_cfg.sampling_rate,
            format="FLAC"
        )

        set_progress(job_id, 70)

        # 4) Compose final video with audio
        output_path = job_folder / "output.mp4"
        make_video(video_info, output_path, audio, sampling_rate=seq_cfg.sampling_rate)

        set_progress(job_id, 100)
        set_status(job_id, "done")
        log.info(f"[JOB {job_id}] Completed -> {output_path}")

    except Exception as e:
        set_status(job_id, f"error: {e}")
        log.exception(f"[JOB {job_id}] ERROR: {e}")


def main():
    net, feature_utils, seq_cfg, model_cfg = load_model()

    log.info("Worker started. Waiting for jobs...")

    while True:
        # read queue
        if QUEUE_FILE.exists():
            try:
                queue = json.loads(QUEUE_FILE.read_text())
            except Exception:
                queue = []
        else:
            queue = []

        if queue:
            job = queue.pop(0)
            QUEUE_FILE.write_text(json.dumps(queue, indent=2))

            job_id = job["job_id"]
            log.info(f"[JOB {job_id}] Taken from queue.")
            process_job(job_id, job, net, feature_utils, seq_cfg)

        time.sleep(1)


if __name__ == "__main__":
    main()

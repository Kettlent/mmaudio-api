import os
import json
import time
import torch
import torchaudio
from pathlib import Path
from mmaudio.eval_utils import load_video, generate, make_video
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils
from mmaudio.eval_utils import all_model_cfg

JOBS_DIR = Path("/workspace/mmaudio-jobs")
INCOMING = JOBS_DIR / "incoming"
DONE = JOBS_DIR / "done"
LOGS = JOBS_DIR / "logs"

VARIANT = "large_44k_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

def load_model():
    model_cfg = all_model_cfg[VARIANT]
    model_cfg.download_if_needed()
    seq_cfg = model_cfg.seq_cfg

    net = get_my_mmaudio(model_cfg.model_name).to(DEVICE, DTYPE).eval()
    weights = torch.load(model_cfg.model_path, map_location=DEVICE, weights_only=True)
    net.load_weights(weights)

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model_cfg.vae_path,
        synchformer_ckpt=model_cfg.synchformer_ckpt,
        mode=model_cfg.mode,
        enable_conditions=True,
        bigvgan_vocoder_ckpt=model_cfg.bigvgan_16k_path,
        need_vae_encoder=False
    ).to(DEVICE, DTYPE).eval()

    return net, feature_utils, seq_cfg, model_cfg


net, feature_utils, seq_cfg, model_cfg = load_model()
print("Worker loaded model successfully.")

def process_job(job_file):
    job_data = json.loads(job_file.read_text())

    job_id = job_data["job_id"]
    prompt = job_data["prompt"]
    negative = job_data["negative_prompt"]
    duration = job_data["duration"]
    cfg_strength = job_data["cfg_strength"]
    num_steps = job_data["num_steps"]
    video_path = Path(job_data["video_path"])

    out_video_path = DONE / f"{job_id}.mp4"
    audio_path = DONE / f"{job_id}.flac"

    try:
        # Load video
        video_info = load_video(video_path, duration)
        seq_cfg.duration = video_info.duration_sec
        net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

        rng = torch.Generator(device=DEVICE).manual_seed(42)
        fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

        clip = video_info.clip_frames.unsqueeze(0)
        sync = video_info.sync_frames.unsqueeze(0)

        with torch.no_grad():
            audios = generate(
                clip, sync, [prompt],
                negative_text=[negative],
                feature_utils=feature_utils,
                net=net, fm=fm, rng=rng,
                cfg_strength=cfg_strength
            )

        audio = audios.float().cpu()[0]
        torchaudio.save(str(audio_path), audio.unsqueeze(0), seq_cfg.sampling_rate, format="FLAC")

        make_video(video_info, out_video_path, audio, sampling_rate=seq_cfg.sampling_rate)

        (DONE / f"{job_id}.json").write_text(json.dumps({"status": "done"}))

        print(f"[Worker] Job {job_id} complete.")

    except Exception as e:
        print("Worker error:", e)
        (DONE / f"{job_id}.json").write_text(json.dumps({"status": "error", "detail": str(e)}))


def main():
    print("Worker started. Watching for jobs...")
    while True:
        for job_file in INCOMING.glob("*.json"):
            process_job(job_file)
            os.remove(job_file)
        time.sleep(1)

if __name__ == "__main__":
    main()

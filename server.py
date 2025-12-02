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

# curl -X POST "https://06pk065k5dkotm-8080.proxy.runpod.net/generate_video" \
#   -F "prompt=" \
#   -F "negative_prompt=" \
#   -F "video=@/Users/scallercell_2/Downloads/silencevideo.mp4"



# server.py
import json
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import anyio

BASE_DIR = Path("/workspace/mmaudio-api")
JOB_DIR = BASE_DIR / "jobs"
QUEUE_FILE = BASE_DIR / "queue.json"

JOB_DIR.mkdir(exist_ok=True)

app = FastAPI(title="MMAudio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


def enqueue_job(data):
    job_id = str(uuid.uuid4())
    folder = JOB_DIR / job_id
    folder.mkdir(exist_ok=True)

    data["job_id"] = job_id

    if QUEUE_FILE.exists():
        queue = json.loads(QUEUE_FILE.read_text())
    else:
        queue = []

    queue.append(data)
    QUEUE_FILE.write_text(json.dumps(queue, indent=2))
    return job_id


@app.post("/generate_video")
async def generate_video(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    duration: float = Form(8.0),
    cfg_strength: float = Form(4.5),
    num_steps: int = Form(25),
    video: UploadFile = File(...)
):

    job_id = str(uuid.uuid4())
    folder = JOB_DIR / job_id
    folder.mkdir(exist_ok=True)

    input_video_path = folder / "input.mp4"
    with open(input_video_path, "wb") as f:
        f.write(await video.read())

    job_data = dict(
        job_id=job_id,
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration=duration,
        cfg_strength=cfg_strength,
        num_steps=num_steps,
        input_video=str(input_video_path),
    )

    if QUEUE_FILE.exists():
        queue = json.loads(QUEUE_FILE.read_text())
    else:
        queue = []
    queue.append(job_data)
    QUEUE_FILE.write_text(json.dumps(queue, indent=2))

    return {"job_id": job_id}


@app.get("/progress/{job_id}")
async def progress(job_id: str):
    progress_file = JOB_DIR / job_id / "progress.txt"

    async def streamer():
        last_value = None
        while True:
            if progress_file.exists():
                value = progress_file.read_text()
                if value != last_value:
                    yield f"data: {value}\n\n"
                    last_value = value
            await anyio.sleep(0.5)

    return StreamingResponse(streamer(), media_type="text/event-stream")


@app.get("/status/{job_id}")
def status(job_id: str):
    status_file = JOB_DIR / job_id / "status.txt"
    if not status_file.exists():
        return {"status": "pending"}
    return {"status": status_file.read_text()}


@app.get("/result/{job_id}")
def result(job_id: str):
    output_video = JOB_DIR / job_id / "output.mp4"
    if not output_video.exists():
        return {"error": "not_ready"}

    return FileResponse(
        str(output_video),
        media_type="video/mp4",
        filename=f"{job_id}.mp4",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.mp4"'}
    )


@app.get("/")
def root():
    return {"status": "mmaudio server running"}


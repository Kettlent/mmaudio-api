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
import uuid
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse

JOB_DIR = Path("./jobs")
JOB_DIR.mkdir(exist_ok=True)

app = FastAPI()

QUEUE_PATH = JOB_DIR / "queue.json"
if not QUEUE_PATH.exists():
    QUEUE_PATH.write_text("[]")


def add_job(job):
    queue = json.loads(QUEUE_PATH.read_text())
    queue.append(job)
    QUEUE_PATH.write_text(json.dumps(queue))


@app.post("/generate_video")
async def generate_video(
    video: UploadFile = File(...),
    prompt: str = Form(...),
    duration: float = Form(8.0),
    cfg_strength: float = Form(4.5),
    num_steps: int = Form(25),
):
    job_id = str(uuid.uuid4())

    input_path = JOB_DIR / f"{job_id}_input.mp4"
    input_path.write_bytes(await video.read())

    add_job({
        "job_id": job_id,
        "prompt": prompt,
        "duration": duration,
        "cfg_strength": cfg_strength,
        "num_steps": num_steps,
        "input_path": str(input_path)
    })

    return {"job_id": job_id}


@app.get("/progress/{job_id}")
async def progress(job_id: str):
    """ SSE stream of progress """
    async def event_stream():
        progress_file = JOB_DIR / f"{job_id}_progress.txt"
        status_file = JOB_DIR / f"{job_id}_status.txt"

        last_value = None

        while True:
            if status_file.exists():
                status = status_file.read_text()
                if "done" in status:
                    yield f"data: 100\n\n"
                    break
                if "error" in status:
                    yield f"data: error\n\n"
                    break

            if progress_file.exists():
                pct = progress_file.read_text()
                if pct != last_value:
                    last_value = pct
                    yield f"data: {pct}\n\n"

            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/result/{job_id}")
def result(job_id: str):
    file_path = JOB_DIR / f"{job_id}_output.mp4"
    if file_path.exists():
        return FileResponse(file_path, media_type="video/mp4", filename="generated.mp4")
    return {"error": "not_ready"}

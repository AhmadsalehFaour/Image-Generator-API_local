"""
Local Image Generation API (FastAPI) — Ollama-like
"""
import os
import io
import uuid
import time
import json
import queue
import threading
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Literal

import torch
from fastapi import FastAPI, Depends, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# ===== Settings =====
class Settings(BaseSettings):
    MODEL_DIR: str = os.path.abspath(os.getenv("MODEL_DIR", "./models"))
    OUTPUT_DIR: str = os.path.abspath(os.getenv("OUTPUT_DIR", "./outputs"))
    DEVICE: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "1"))  # 1 is safer for 8GB VRAM
    API_KEY: Optional[str] = os.getenv("API_KEY")  # if set, X-API-Key is required

settings = Settings()
os.makedirs(settings.MODEL_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

# ===== Simple API key auth via header =====
class APIKeyAuth:
    def __call__(self, x_api_key: Optional[str] = Header(default=None)):
        if settings.API_KEY and x_api_key != settings.API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return True

auth_dep = APIKeyAuth()

# ===== Model utilities =====
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)

try:
    from diffusers import FluxPipeline  # requires recent diffusers
    HAS_FLUX = True
except Exception:
    FluxPipeline = None
    HAS_FLUX = False


def preferred_dtype():
    try:
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    except Exception:
        return torch.float16


def build_scheduler(pipe, name: str):
    name = (name or "dpmpp").lower()
    if name in {"dpmpp", "dpmpp2m", "dpmpp_2m", "dpmpp-karras"}:
        return DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


class ModelManager:
    def __init__(self, device: str = settings.DEVICE):
        self.device = device
        self.pipes: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def list_loaded(self) -> Dict[str, str]:
        with self.lock:
            return {k: type(v).__name__ for k, v in self.pipes.items()}

    def unload(self, name: str) -> bool:
        with self.lock:
            if name in self.pipes:
                try:
                    del self.pipes[name]
                    if self.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    return True
                except Exception:
                    return False
            return False

    def load(self, name: Literal["sd15", "flux"], local_dir: Optional[str] = None, scheduler: Optional[str] = None):
        dtype = preferred_dtype()
        with self.lock:
            if name == "sd15":
                model_path = local_dir or os.path.join(settings.MODEL_DIR, "sd15")
                pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
                pipe.scheduler = build_scheduler(pipe, scheduler or "dpmpp")
            elif name == "flux":
                if not HAS_FLUX:
                    raise HTTPException(status_code=500, detail="FluxPipeline not available; upgrade diffusers")
                model_path = local_dir or os.path.join(settings.MODEL_DIR, "flux-schnell")
                pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype)
            else:
                raise HTTPException(status_code=400, detail="Unsupported model name")

            pipe = pipe.to(self.device)
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            if hasattr(pipe, "enable_attention_slicing"):
                try:
                    pipe.enable_attention_slicing()
                except Exception:
                    pass
            if hasattr(pipe, "set_progress_bar_config"):
                pipe.set_progress_bar_config(disable=True)

            self.pipes[name] = pipe
            return {"name": name, "device": self.device, "dtype": str(dtype), "scheduler": type(pipe.scheduler).__name__ if hasattr(pipe, "scheduler") else None}

    def get(self, name: str):
        with self.lock:
            pipe = self.pipes.get(name)
            if pipe is None:
                raise HTTPException(status_code=400, detail=f"Model '{name}' not loaded")
            return pipe


model_manager = ModelManager()

# ===== Pydantic models =====
class LoadModelBody(BaseModel):
    name: Literal["sd15", "flux"]
    local_dir: Optional[str] = Field(default=None, description="local model path")
    scheduler: Optional[str] = Field(default=None, description="dpmpp (default), ...")

class UnloadModelBody(BaseModel):
    name: Literal["sd15", "flux"]

class Txt2ImgBody(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    model: Literal["sd15", "flux"] = "sd15"
    width: int = 512
    height: int = 512
    steps: int = 20
    guidance: float = 7.5
    seed: Optional[int] = None
    scheduler: Optional[str] = None
    filename: Optional[str] = None

class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "error"]
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    error: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

# ===== Job system =====
class Job:
    def __init__(self, req: Txt2ImgBody):
        self.id = uuid.uuid4().hex
        self.req = req
        self.status: Literal["queued", "running", "done", "error"] = "queued"
        self.image_path: Optional[str] = None
        self.error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None


job_queue: "queue.Queue[Job]" = queue.Queue()
job_registry: Dict[str, Job] = {}


def ensure_workers():
    for i in range(settings.MAX_WORKERS):
        t = threading.Thread(target=worker_loop, name=f"worker-{i}", daemon=True)
        t.start()


def save_image(image, filename: Optional[str] = None) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    base = filename or f"{ts}-{uuid.uuid4().hex[:8]}.png"
    out_path = os.path.join(settings.OUTPUT_DIR, base)
    image.save(out_path)
    return out_path


def generate(pipe, req: Txt2ImgBody):
    g = torch.Generator(device=settings.DEVICE if settings.DEVICE != "cpu" else "cpu")
    if req.seed is not None:
        g = g.manual_seed(req.seed)

    kwargs = dict(
        prompt=req.prompt,
        width=req.width,
        height=req.height,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance,
        generator=g,
    )
    if req.negative_prompt is not None:
        kwargs["negative_prompt"] = req.negative_prompt

    if hasattr(pipe, "scheduler") and req.scheduler:
        pipe.scheduler = build_scheduler(pipe, req.scheduler)

    image = pipe(**kwargs).images[0]
    return image


def worker_loop():
    while True:
        job: Job = job_queue.get()
        job.status = "running"
        job.started_at = time.time()
        try:
            pipe = model_manager.get(job.req.model)
            image = generate(pipe, job.req)
            path = save_image(image, job.req.filename)
            job.image_path = path
            job.status = "done"
        except Exception as e:
            logging.exception("Job failed")
            job.status = "error"
            job.error = str(e)
        finally:
            job.finished_at = time.time()
            job_queue.task_done()


# ===== FastAPI app =====
app = FastAPI(title="Local Image API (Ollama-like)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=settings.OUTPUT_DIR), name="static")


@app.get("/health")
def health():
    return {"ok": True, "device": settings.DEVICE}


@app.get("/models", dependencies=[Depends(auth_dep)])
def list_models():
    return {"loaded": model_manager.list_loaded(), "model_dir": settings.MODEL_DIR}


@app.post("/models/load", dependencies=[Depends(auth_dep)])
def load_model(body: LoadModelBody):
    info = model_manager.load(name=body.name, local_dir=body.local_dir, scheduler=body.scheduler)
    return {"status": "ok", "info": info}


@app.post("/models/unload", dependencies=[Depends(auth_dep)])
def unload_model(body: UnloadModelBody):
    ok = model_manager.unload(body.name)
    return {"status": "ok" if ok else "not_loaded"}


@app.post("/generate/txt2img", response_model=JobStatus, dependencies=[Depends(auth_dep)])
def txt2img(body: Txt2ImgBody, async_mode: bool = Query(False, description="true to return job_id instead of waiting")):
    if body.width * body.height > 1024 * 1024 and body.steps > 30:
        raise HTTPException(status_code=400, detail="Use lower resolution or fewer steps")

    if not async_mode:
        pipe = model_manager.get(body.model)
        try:
            image = generate(pipe, body)
            path = save_image(image, body.filename)
            return JobStatus(
                id=uuid.uuid4().hex,
                status="done",
                image_path=path,
                image_url=f"/static/{os.path.basename(path)}",
                meta={"model": body.model, "w": body.width, "h": body.height, "steps": body.steps}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    job = Job(body)
    job_registry[job.id] = job
    job_queue.put(job)
    return JobStatus(id=job.id, status=job.status)


@app.get("/jobs/{job_id}", response_model=JobStatus, dependencies=[Depends(auth_dep)])
def job_status(job_id: str):
    job = job_registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    url = f"/static/{os.path.basename(job.image_path)}" if job.image_path else None
    return JobStatus(
        id=job.id,
        status=job.status,
        image_path=job.image_path,
        image_url=url,
        error=job.error,
        meta={"started_at": job.started_at, "finished_at": job.finished_at}
    )


ensure_workers()

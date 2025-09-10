# Local Image Generation API (Ollama-like)

A **FastAPI-based REST API** for local image generation using AI models such as **Stable Diffusion 1.5** and **FLUX**.
The project is inspired by **Ollama**, but focused on generating **images instead of text**.

---

## ✨ Features

- 🚀 **Fast & Local** — Run models entirely on your machine using **FastAPI + Docker**.
- 🔄 **Dynamic Model Management** — Load and unload models on demand via API endpoints.
- 🎨 **Text-to-Image (`txt2img`)** — Generate images with customizable parameters:
  - Prompt & Negative Prompt
  - Width & Height
  - Inference Steps
  - Guidance Scale
  - Seed for reproducibility
- 🔑 **API Key Authentication** — Protect your endpoints with simple header-based auth.
- ⚡ **Job Queue System** — Support for async or queued image generation.
- 📦 **Model Management** — Use local models or automatically download from Hugging Face.

---

## 🛠 Requirements

- **Docker** & **Docker Compose**
- NVIDIA GPU with CUDA support (recommended)
- Python 3.10+ (if running without Docker)

---

## 🚀 Quickstart

Clone the repo and set up the environment:

```bash
git clone https://github.com/your-username/image-generator-api.git
cd image-generator-api

cp .env.example .env
mkdir -p models outputs
docker compose build
docker compose up -d
```

Check health:

```bash
curl http://localhost:7878/health
```

---

## 📥 Model Management

### Load a Model
```bash
curl -X POST http://localhost:7878/models/load -H "Content-Type: application/json" \
  -d '{"name":"sd15","local_dir":"/models/sd15"}'
```

### Unload a Model
```bash
curl -X POST http://localhost:7878/models/unload -H "Content-Type: application/json" \
  -d '{"name":"sd15"}'
```

### List Loaded Models
```bash
curl http://localhost:7878/models -H "X-API-Key: your_api_key"
```

---

## 🎨 Image Generation

### Generate an Image
```bash
curl -X POST http://localhost:7878/generate/txt2img -H "Content-Type: application/json" \
  -d '{"prompt":"desert sunset, cinematic","model":"sd15","width":512,"height":512,"steps":20,"guidance":7.5}'
```

The API will return an `image_url` you can open in your browser.

### Async Mode (Jobs)
You can generate in async mode:
```bash
curl -X POST "http://localhost:7878/generate/txt2img?async_mode=true" -H "Content-Type: application/json" \
  -d '{"prompt":"futuristic city, neon lights","model":"sd15"}'
```

Check job status:
```bash
curl http://localhost:7878/jobs/<job_id>
```

---

## ⚙️ Configuration

All configuration is handled via the `.env` file:

```env
# Ports
HOST_PORT=7878
APP_PORT=7878

# Auth
API_KEY=change-me

# Runtime
device=cuda
MAX_WORKERS=1

# Model Repos
SD15_REPO=runwayml/stable-diffusion-v1-5
SD15_DIR=/models/sd15
FLUX_REPO=black-forest-labs/FLUX.1-schnell
FLUX_DIR=/models/flux-schnell
```

---

## 📦 Dependencies

Main Python dependencies:
- `fastapi`
- `uvicorn`
- `pydantic`
- `transformers`
- `diffusers`
- `accelerate`
- `huggingface_hub`

See [requirements.txt](./requirements.txt) for the full list.

---

## 📸 Example Output

| Prompt | Generated Image |
|--------|----------------|
| `desert sunset, cinematic` | ![Sample](./examples/sample.png) |

---

## 🛡 Security

- Use **API_KEY** in `.env` to require `X-API-Key` header for all API requests.
- Example:
```bash
curl http://localhost:7878/models -H "X-API-Key: change-me"
```

---

## 🤝 Contributing

Contributions are welcome! Please fork this repo and submit a PR.

---

## 📄 License

MIT License © 2025


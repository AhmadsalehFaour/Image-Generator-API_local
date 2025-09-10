# Local Image Generation API (Ollama-like)

## Quickstart
```bash
cp .env.example .env
mkdir -p models outputs
docker compose build
docker compose up -d
curl http://localhost:7777/health
```

Load a model (first time):
```bash
curl -X POST http://localhost:7777/models/load -H "Content-Type: application/json" \
  -d '{"name":"sd15","local_dir":"/models/sd15"}'
```

Generate:
```bash
curl -X POST http://localhost:7777/generate/txt2img -H "Content-Type: application/json" \
  -d '{"prompt":"desert sunset, cinematic","model":"sd15","width":512,"height":512,"steps":20,"guidance":7.5}'
```
Open the returned `image_url` in your browser.

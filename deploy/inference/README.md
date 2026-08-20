# Inference ops (GPU server 2)

Diagnostics for the current Ollama, Hugging Face downloads, four vLLM containers, and before/after benches.

**Read this first:** [GUIDE.md](GUIDE.md) — what to run, how to read the output, when to reindex, how to roll back.

Quick start after you have SSH to server 2:

```bash
cd deploy/inference
cp .env.example .env          # edit HF_HOME / GPU_ID if needed
./scripts/ollama_diag.sh
./scripts/ollama_pipeline_trace.sh
```

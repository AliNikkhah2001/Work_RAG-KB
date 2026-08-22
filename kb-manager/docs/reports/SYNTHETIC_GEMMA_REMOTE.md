# Synthetic Generation — Remote 2×GPU Gemma 30B Execution Guide

**Branch:** `feat/synthetic-gemma-remote` — **Not merged to master**. Pull this branch on your remote machine. No local Gemma code edits — Gemma is pure black-box API via `KB_LLM_BACKEND`.

## Prerequisites (Remote Machine)
- 2× GPU with 24GB+ VRAM each (total 48GB)
- Docker + `vllm` **or** `ollama`
- Python 3.11+, `pip install datasets pyyaml`
- Pull this branch: `git clone -b feat/synthetic-gemma-remote https://github.com/AliNikkhah2001/Work_RAG-KB.git`

## Option A: Ollama (Easiest, 1 command)
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma2:27b
ollama serve &  # http://localhost:11434
export KB_LLM_BACKEND=ollama
export KB_LLM_MODEL=gemma2:27b
export OLLAMA_BASE_URL=http://localhost:11434
```

## Option B: vLLM (Higher throughput, tensor parallel)
```bash
docker run --gpus all -p 8000:8000 --ipc=host \
  vllm/vllm-openai:latest \
  --model google/gemma-2-27b-it \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --max-model-len 4096
export KB_LLM_BACKEND=vllm
export KB_LLM_MODEL=google/gemma-2-27b-it
export KB_LLM_BASE_URL=http://localhost:8000/v1
```

## Verify Black-Box API (Mock vs Real)
```bash
# Mock (no GPU): returns [{"query":"تست","type":"verbatim"}]
KB_LLM_BACKEND=mock python -c "from kb_manager.llm import create_llm_client_from_config; from kb_manager.config import load_config; c=create_llm_client_from_config(load_config()); print(c.generate('سلام').text)"

# Real Ollama: should return Persian QA JSON
KB_LLM_BACKEND=ollama KB_LLM_MODEL=gemma2:27b python -c "from kb_manager.llm import create_llm_client; c=create_llm_client('ollama','gemma2:27b'); print(c.generate('سلام').text[:200])"
```

## Run Synthetic Generation
```bash
cd kb-manager

# Smoke test (10 chunks, no validation)
python synthetic_generation/run_generation.py --config synthetic_generation/config.yaml --db-path data/kb_test.db --output-dir synthetic_generation/output --limit 10 --skip-validation
cat synthetic_generation/output/synthetic_qa.jsonl | head -n 2

# Full run (all 6 types, 30% conversational, threshold 0.7)
python synthetic_generation/run_generation.py --config synthetic_generation/config.yaml --db-path data/kb_test.db --output-dir synthetic_generation/output --limit 100
# or full KB
python synthetic_generation/run_generation.py --config synthetic_generation/config.yaml --db-path data/kb_test.db --output-dir synthetic_generation/output

# Outputs
ls -lh synthetic_generation/output/
# synthetic_qa.jsonl (50K QA target)
# synthetic_conversations.jsonl (15K conv)
# generation_metadata.json
```

## Prompts Used (FaMTEB Methodology, `synthetic_generation/prompts/`)
- `qa_generation.txt` — 6 types: verbatim, paraphrase, conversational (می‌شه…), typo (می‌شود→میشه), keyword_only (امتیاز اعتباری…), reworded
- `conversational.txt` — 3-turn with coreference (`برای حقوقی هم همینه؟`) + ellipsis
- `typo_injection.txt` — 13 Persian typo rules, ZWNJ drop (currently orphaned, HyDE will use)
- `keyword_extraction.txt` — domain keywords (currently orphaned)
- `validation.txt` — LLM-as-judge 4 scores (relevance, accuracy, naturalness, diversity) mean≥0.7

## Validation & Human Audit
- Auto: `SyntheticValidator` in `generators/validator.py` uses **same Gemma** as judge (correlated error noted) — `threshold 0.7`
- Human: `config.yaml: human_audit_samples: 100` — randomly sample 100, manual Persian check

## Push Back to GitHub
```bash
# On remote, after generation
git add synthetic_generation/output/
git commit -m "data: synthetic Gemma 30B generation (50K QA)"
git push origin feat/synthetic-gemma-remote
# Or SCP back to local machine
```

## Troubleshooting
- `KB_LLM_BACKEND` not set → silent `MockLLMClient` → check `echo $KB_LLM_BACKEND`
- OOM → reduce `config.yaml: generation.max_chunk_length 2000→1000` or `vllm --gpu-memory-utilization 0.85`
- Low pass rate → lower `validation.threshold 0.7→0.65` or fix prompts

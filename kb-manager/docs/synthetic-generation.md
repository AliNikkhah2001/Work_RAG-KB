# Synthetic Data Generation with Gemma 30B

Complete guide for generating synthetic Persian QA and conversational data using Gemma-2-27B on dual-GPU setup.

---

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| **GPUs** | 2× NVIDIA GPU with 24GB+ VRAM each (e.g., 2× RTX 3090/4090 or A100) |
| **VRAM Total** | 48GB+ (for Gemma-2-27B 4-bit AWQ) |
| **System RAM** | 64GB+ recommended |
| **Storage** | 200GB+ NVMe for model cache + output |

---

## Software Setup

### 1. Install Dependencies

```bash
# On the GPU machine
pip install vllm==0.5.3.post1  # For tensor parallelism
pip install ollama             # Alternative: Ollama
pip install pyyaml datasets transformers accelerate
pip install -e /path/to/kb-manager  # Install KB Manager in editable mode
```

### 2. Model Setup

**Option A: Ollama (Easier setup)**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Gemma-2-27B
ollama pull gemma2:27b

# Verify
ollama run gemma2:27b "سلام"
```

**Option B: vLLM (Higher throughput)**
```bash
# Start vLLM server with tensor parallelism
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-27b-it \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --port 8000
```

---

## Configuration

Edit `synthetic_generation/config.yaml`:

```yaml
model:
  name: "gemma2:27b"              # Model name
  backend: "ollama"               # ollama | vllm
  base_url: "http://localhost:11434"
  tensor_parallel_size: 2
  max_tokens: 1024
  temperature: 0.4

generation:
  num_samples_per_chunk: 8
  conversation_ratio: 0.3
  query_types:
    verbatim: 0.15
    paraphrase: 0.20
    conversational: 0.20
    typo: 0.15
    keyword_only: 0.15
    reworded: 0.15

validation:
  enabled: true
  threshold: 0.7
  validator_model: "gemma2:27b"
```

---

## Running Generation

### 1. Prepare KB Database

```bash
# Ensure KB database exists
cd /path/to/kb-manager
KB_DB_URL="sqlite+aiosqlite:///data/kb_test.db" python -c "
from kb_manager.web.app import db
from kb_manager.models.database import Chunk
import asyncio
async def check():
    async with db.session() as s:
        from sqlalchemy import select, func
        c = await s.execute(select(func.count(Chunk.id)).where(Chunk.chunk_type=='qa_pair'))
        print(f'QA pairs: {c.scalar()}')
asyncio.run(check())
"
```

### 2. Run Generation

```bash
cd /path/to/kb-manager

# Basic run (process all chunks)
python synthetic_generation/run_generation.py \
  --config synthetic_generation/config.yaml \
  --db-path data/kb_test.db \
  --output-dir synthetic_generation/output

# Limited test run
python synthetic_generation/run_generation.py \
  --config synthetic_generation/config.yaml \
  --db-path data/kb_test.db \
  --output-dir synthetic_generation/output \
  --limit 10

# Skip validation (faster)
python synthetic_generation/run_generation.py \
  --config synthetic_generation/config.yaml \
  --db-path data/kb_test.db \
  --output-dir synthetic_generation/output \
  --skip-validation
```

### 3. Monitor Progress

```bash
# Watch logs
tail -f synthetic_generation/output/generation.log

# Check output files
ls -la synthetic_generation/output/
```

---

## Output Files

| File | Description |
|------|-------------|
| `synthetic_qa.jsonl` | QA pairs (one per line) |
| `synthetic_conversations.jsonl` | Multi-turn conversations |
| `generation_metadata.json` | Generation stats and config |
| `generation.log` | Detailed logs |

### QA Pair Format (JSONL)

```json
{
  "query": "امتیاز اعتباری چطور محاسبه می‌شود؟",
  "type": "verbatim",
  "chunk_id": "abc123...",
  "expected_chunk_id": "abc123...",
  "validation": {
    "overall_score": 0.87,
    "relevance": 0.9,
    "accuracy": 0.85,
    "naturalness": 0.88,
    "diversity": 0.85
  }
}
```

### Conversation Format (JSONL)

```json
{
  "conversation": [
    {"role": "user", "content": "سلام، امتیاز اعتباری چیه؟"},
    {"role": "assistant", "content": "امتیاز اعتباری یک عدد است که..."},
    {"role": "user", "content": "برای حقوقی هم همینه؟"},
    {"role": "assistant", "content": "بله، مدل‌های حقوقی و حقیقی هر دو..."},
    {"role": "user", "content": "مشخصات مدل حقیقی رو بگو"}
  ],
  "target_chunk_id": "abc123...",
  "topic": "امتیاز اعتباری",
  "coreference_used": true,
  "ellipsis_used": true,
  "num_turns": 3
}
```

---

## Validation System

The system uses **LLM-as-judge** (Gemma-2-27B) for quality control:

| Metric | Description | Weight |
|--------|-------------|--------|
| **Relevance** | Semantic match to target chunk | 25% |
| **Accuracy** | Factual correctness | 25% |
| **Naturalness** | Natural Persian phrasing | 25% |
| **Diversity** | Unique vs other samples | 25% |

**Threshold**: Overall score ≥ 0.7 required to pass

---

## Performance Tuning

### For Higher Throughput

```yaml
# In config.yaml
model:
  tensor_parallel_size: 2  # Use both GPUs
  max_tokens: 1024
  
generation:
  batch_size: 4  # Process multiple chunks in parallel (requires code changes)
```

### For Higher Quality

```yaml
model:
  temperature: 0.3  # More deterministic
  
validation:
  threshold: 0.75  # Stricter threshold
```

---

## Expected Output Stats

| Metric | Target |
|--------|--------|
| QA pairs per chunk | 6-8 |
| Conversation ratio | 30% |
| Validation pass rate | 70-85% |
| Throughput | ~10-20 chunks/min (2×24GB GPU) |
| Total time (6K chunks) | 4-8 hours |

---

## Troubleshooting

### OOM Errors
```bash
# Reduce tensor parallel or use 8-bit quantization
# In vLLM: --gpu-memory-utilization 0.85
```

### Slow Generation
```bash
# Use vLLM instead of Ollama
# Increase tensor_parallel_size
# Reduce max_tokens
```

### Low Validation Pass Rate
```yaml
# Adjust prompts or lower threshold
validation:
  threshold: 0.65
```

---

## Using Generated Data

### 1. Add to Benchmark

```python
from kb_manager.famteb import save_benchmark_dataset

# Convert to benchmark format
samples = []
for qa in qa_data:
    samples.append({
        "query": qa["query"],
        "expected_chunk_ids": [qa["expected_chunk_id"]],
        "format": qa["type"],
        "difficulty": "medium",
        "category": "factual",
    })

save_benchmark_dataset(samples, "data/synthetic_benchmark.json")
```

### 2. Run Benchmark

```bash
KB_DB_URL="sqlite+aiosqlite:///data/kb_test.db" \
python run_benchmark.py synthetic_benchmark.json 5
```

### 3. Fine-tune Reranker

```python
# Use synthetic data to fine-tune cross-encoder
from sentence_transformers import CrossEncoder
model = CrossEncoder("microsoft/mdeberta-v3-base-xsmall", num_labels=1)
model.fit(train_dataloader, epochs=3)
```

---

## Quality Checklist

- [ ] All 6 query types generated
- [ ] Conversational samples have coreference/ellipsis
- [ ] Typo samples have realistic Persian errors
- [ ] Keyword-only samples contain only domain terms
- [ ] Validation pass rate > 70%
- [ ] Human audit of 100 samples > 90% quality
- [ ] No duplicate queries
- [ ] All chunks covered
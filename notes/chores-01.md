# Chores-01

Discussions and notes on various chores in github compatible markdown.
There is also a [todo.md](todo.md) file and it tracks tasks and in
general there should be a chore section for each task with the why
and how this task will be completed.

See [Chores format](README.md#chores-format)

## How Karpathy's autoresearch works (20260322 0.1.0)

### Concept

Autoresearch is an autonomous AI research loop. You give an AI coding
agent (Claude, Codex, etc.) a small but real LLM training setup and
let it experiment on its own — modifying code, training for 5 minutes,
checking if results improved, keeping or discarding, and repeating
indefinitely. The human's role shifts from writing training code to
writing `program.md` — a markdown file that instructs the agent on
what to try and how to behave.

The tagline: you sleep, the agent runs ~100 experiments overnight,
you wake up to a log of results and (hopefully) a better model.

### The three-file design

The repo is intentionally minimal — three files that matter:

| File | Who edits | Purpose |
|------|-----------|---------|
| `prepare.py` | Nobody (read-only) | Data download, BPE tokenizer training, dataloader, evaluation metric |
| `train.py` | The AI agent | GPT model, optimizer, hyperparameters, training loop |
| `program.md` | The human | Agent instructions — the "research org code" |

This separation is key: the agent only touches `train.py`, keeping
diffs reviewable. The human steers research direction via `program.md`
without touching Python.

### The model (`train.py`)

A single-GPU GPT implementation cherry-picked from Karpathy's
[nanochat](https://github.com/karpathy/nanochat), using several
modern techniques:

**Architecture:**
- **Transformer blocks** with pre-norm (RMSNorm via `F.rms_norm`)
- **Rotary position embeddings (RoPE)** — precomputed cos/sin buffers,
  applied to Q and K before attention, with QK-norm
- **Grouped query attention** — supports `n_kv_head < n_head` (though
  defaults to `n_kv_head == n_head`)
- **Flash Attention 3** — via the `kernels` package; selects
  `varunneal/flash-attention-3` on Hopper (H100), falls back to
  `kernels-community/flash-attn3` on other GPUs
- **Sliding window attention** — configurable pattern string (e.g.
  `"SSSL"` = 3 short-window layers then 1 full-context layer, repeated).
  Short window = half the sequence length. Last layer always full context
- **Value embeddings (ResFormer)** — on alternating layers, a separate
  embedding table produces value residuals mixed into V via a learned
  input-dependent gate per head: `v = v + 2*sigmoid(gate(x[:32])) * ve`
- **Residual scaling** — per-layer learnable `resid_lambda` and
  `x0_lambda` scalars that mix the residual stream with the original
  embedding: `x = lambda*x + x0_lambda*x0`
- **Squared ReLU MLP** — `relu(x)^2` activation, 4x expansion ratio
- **Logit softcapping** — `15 * tanh(logits/15)` before the final
  cross-entropy, bounding logit magnitudes (technique from Gemma 2)

**Default hyperparameters:**
- `DEPTH = 8` layers, `ASPECT_RATIO = 64` → `model_dim = 512`,
  `HEAD_DIM = 128` → 4 heads
- `TOTAL_BATCH_SIZE = 2^19` (~524K tokens per step)
- `DEVICE_BATCH_SIZE = 128`, `MAX_SEQ_LEN = 2048`
- ~50M parameters at default depth

### The optimizer (`MuonAdamW`)

A hybrid optimizer that uses different algorithms for different
parameter types:

**Muon** (for 2D matrix parameters — attention projections, MLP weights):
- Nesterov momentum with warmup (0.85 → 0.95 over 300 steps)
- **Polar Express orthogonalization** — approximates the matrix polar
  decomposition (nearest orthogonal matrix) using 5 Newton-Schulz
  iterations with hardcoded polynomial coefficients. This is the core
  of Muon: it steepest-descends in the space of orthogonal matrices
- **NorMuon variance reduction** — normalizes gradient variance using
  a second-moment EMA, applied per-row or per-column depending on
  matrix shape
- **Cautious weight decay** — only decays weights where the gradient
  and parameter have the same sign
- All operations are `@torch.compile`'d for fusion

**AdamW** (for everything else — embeddings, unembedding, scalars):
- Separate learning rates per group: embeddings (0.6), unembedding
  (0.004), scalars (0.5), all scaled by `1/sqrt(d_model/768)`
- Also `@torch.compile`'d

**Schedules** (all time-based, not step-based):
- LR: optional warmup → constant → linear warmdown over last 50% of
  the time budget
- Weight decay: linear decay from 0.2 to 0 over the time budget
- Muon momentum: linear warmup from 0.85 to 0.95 over 300 steps

### The data pipeline (`prepare.py`)

**Data:** [climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)
— 6543 parquet shards from HuggingFace. By default only 10 training
shards are downloaded (plus 1 pinned validation shard). Each shard
contains a `text` column of documents.

**Tokenizer:** Trained via `rustbpe` (a Rust BPE implementation) using
GPT-4-style split patterns, `vocab_size = 8192` (+ 4 special tokens).
Saved as a tiktoken-compatible pickle.

**Dataloader (`make_dataloader`):**
- BOS-aligned document packing with best-fit bin packing — each row
  of the batch starts with a BOS token, documents are packed to
  minimize wasted space (100% utilization, no padding)
- Uses pinned-memory CPU buffers with async GPU transfer
- Infinite iterator over training shards with epoch tracking

**Evaluation (`evaluate_bpb`):**
- **Bits per byte (BPB)** — the core metric, vocabulary-size-independent
- Sums per-token cross-entropy (nats) over `EVAL_TOKENS` (40 × 524K =
  ~21M tokens), divides by total UTF-8 byte count, converts nats→bits
- Special tokens excluded from both sums
- Uses fixed `MAX_SEQ_LEN = 2048` for comparability

### The agent loop (`program.md`)

The autonomous experiment protocol:

1. **Setup phase** (one-time, with human):
   - Agree on a run tag (e.g. `mar5`), create branch
     `autoresearch/<tag>`
   - Read all in-scope files for context
   - Verify data exists, initialize `results.tsv`
   - Run baseline (unmodified `train.py`) as first experiment

2. **Experiment loop** (runs forever, no human interaction):
   ```
   LOOP FOREVER:
     1. Review git state and past results
     2. Modify train.py with an experimental idea
     3. git commit
     4. Run: uv run train.py > run.log 2>&1
     5. Extract results: grep val_bpb and peak_vram_mb
     6. If crash → read traceback, attempt fix or skip
     7. Log to results.tsv (tab-separated, 5 columns)
     8. If val_bpb improved → keep (advance branch)
     9. If worse → discard (git reset to previous best)
   ```

3. **Key constraints:**
   - Fixed 5-minute time budget per experiment → ~12/hour, ~100/night
   - Only `train.py` is editable; `prepare.py` is read-only
   - No new dependencies allowed
   - Kill runs exceeding 10 minutes
   - **Never stop** — the agent must not ask the human if it should
     continue; it runs until manually interrupted
   - Simplicity criterion: small gains that add ugly complexity aren't
     worth keeping; simplifications that maintain performance are wins

### Rust feasibility — initial observations

A Rust port of autoresearch would need to address several layers:

1. **The agent loop itself** (`program.md` logic) — straightforward to
   port. It's just file I/O, git operations, process spawning, and
   result parsing. Rust excels here.

2. **The training script** (`train.py`) — this is the hard part:
   - Depends heavily on PyTorch (`torch.compile`, autograd, CUDA
     kernels, `torch.amp` autocast)
   - Uses Flash Attention 3 via the `kernels` package (precompiled
     CUDA kernels)
   - The Muon optimizer's polar decomposition is tightly integrated
     with PyTorch's tensor operations and `@torch.compile`
   - Rust ML frameworks (`burn`, `candle`, `tch-rs`) exist but none
     match PyTorch's ecosystem for this use case

3. **The data pipeline** (`prepare.py`) — moderately portable:
   - Parquet reading → `arrow-rs` / `parquet` crates
   - BPE tokenizer → already uses `rustbpe` (a Rust library!)
   - Dataloader with pinned memory and async GPU transfer would need
     custom CUDA interop

4. **Practical approach options:**
   - **Rust orchestrator + Python training**: Port the agent loop and
     experiment management to Rust, keep `train.py` as-is. Lowest
     risk, highest immediate value
   - **Full Rust via `tch-rs`**: Use the PyTorch C++ backend from
     Rust. Gets tensor ops but loses `torch.compile` and most of
     the Python ecosystem
   - **Full Rust via `burn` or `candle`**: Most ambitious. Would need
     to implement Flash Attention, Muon optimizer, and the full model
     from scratch. Educational but significant effort


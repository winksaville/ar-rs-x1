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

## Can autoresearch be ported to Rust? (20260322 0.2.0)

### Answer: Yes, but the scope varies dramatically by approach

The Python code has three distinct layers with different porting
profiles. A full port is feasible with **Burn** as the framework,
but requires implementing the Muon optimizer from scratch. A hybrid
approach (Rust orchestrator + Python training) gives immediate value
with minimal risk.

### Layer-by-layer analysis

#### 1. Agent loop (`program.md` logic) — trivial to port

The experiment orchestration is just file I/O, git/jj operations,
process spawning, log parsing, and TSV writing. Rust excels at all
of this. No ML dependencies needed.

#### 2. Data pipeline (`prepare.py`) — solved in Rust

| Component | Python | Rust equivalent | Status |
|-----------|--------|-----------------|--------|
| Parquet reading | `pyarrow` | `arrow-rs` / `parquet` | Production-grade, used by DataFusion/Polars |
| BPE tokenizer | `rustbpe` + `tiktoken` | `tokenizers` (HuggingFace) or `rustbpe` directly | Already Rust under the hood |
| Dataloader | Custom with pinned memory | Manual via `cudarc` or Burn's data API | Needs custom work for async GPU transfer |
| BPB evaluation | NumPy/PyTorch | Burn or Candle tensor ops | Straightforward |

The data pipeline is the easiest layer — the key crates (`arrow-rs`,
`tokenizers`) are battle-tested and widely used in production.

#### 3. Training script (`train.py`) — the hard part

This is where the real challenge lies. The model uses several modern
techniques that each need a Rust equivalent:

**Feature availability in Rust ML frameworks:**

| Feature | Burn | Candle | tch-rs |
|---------|------|--------|--------|
| Autograd / backprop | Yes (`Autodiff` backend) | Yes | Yes (via libtorch) |
| CUDA backend | Yes (CubeCL) | Yes | Yes |
| Flash Attention | v3 (CUDA, WGPU) | v2 + v3 (CUDA only) | No |
| bf16 mixed precision | Partial (dtype support) | Yes (`to_dtype`) | No (no AMP) |
| RoPE | Built-in `nn::RotaryEncoding` | In model code (LLaMA etc.) | Manual |
| Custom optimizers | Trait-based system | Trait-based system | Limited |
| `torch.compile` | N/A (CubeCL does kernel fusion) | N/A | No |
| Training infra | Strong (TUI dashboard, checkpoints) | Basic | Basic |
| Multi-GPU | Yes (NCCL) | Yes (NCCL) | Yes |

**What does NOT exist in Rust today:**

- **Muon optimizer** — no Rust implementation anywhere. Must be ported
  manually. The core algorithm is Newton-Schulz orthogonalization
  (5 iterations with hardcoded polynomial coefficients) plus Nesterov
  momentum and NorMuon variance reduction. All expressible in standard
  tensor ops, but ~150 lines of non-trivial math to port and validate
- **`torch.compile` equivalent** — Burn's CubeCL provides framework-level
  kernel fusion, which covers some of the same ground but is not
  user-controllable
- **Mature GPT training pipelines** — `femtoGPT` exists (pure Rust,
  OpenCL) but is educational-scale, not production

**Component-by-component porting map for `train.py`:**

| Component | Lines | Porting difficulty | Notes |
|-----------|-------|-------------------|-------|
| `GPTConfig` | ~10 | Trivial | Struct definition |
| `norm` (RMSNorm) | 2 | Easy | `F.rms_norm` → Burn/Candle equivalent |
| `apply_rotary_emb` | 6 | Easy | Burn has built-in `RotaryEncoding` |
| `CausalSelfAttention` | 36 | Medium | Flash Attention available in Burn; value embedding gate logic is custom |
| `MLP` | 10 | Easy | Linear + squared ReLU |
| `Block` / `GPT` | 80 | Medium | Pre-norm residual blocks, `resid_lambda`/`x0_lambda` scaling, window size computation |
| `GPT.forward` | 25 | Medium | Logit softcapping, conditional value embeddings |
| `GPT.init_weights` | 30 | Medium | Custom initialization patterns |
| `MuonAdamW` optimizer | 120 | Hard | Polar Express orthogonalization, NorMuon variance reduction, cautious weight decay. No existing Rust impl |
| `adamw_step_fused` | 8 | Medium | Fused kernel — need Burn's equivalent or manual CUDA |
| `muon_step_fused` | 35 | Hard | Stacked tensor operations, polar decomposition loop |
| Training loop | 60 | Medium | Time-based scheduling, gradient accumulation, GC tricks (N/A in Rust) |
| LR/momentum schedules | 15 | Easy | Pure math |

### Framework recommendation: Burn

**Burn is the strongest candidate** for a full Rust port:

- Flash Attention v3 built-in (matches the Python version)
- RoPE built-in
- Autograd works transparently via `Autodiff` backend decorator
- CubeCL compiles kernels for CUDA, ROCm, Metal, Vulkan, WebGPU —
  the Rust version would run on more hardware than the Python original
- Training infrastructure (dashboards, checkpoints) included
- Active development with growing community

**Candle** is viable if CUDA-only is acceptable and you prefer an API
closer to PyTorch's feel. Less training infrastructure.

**tch-rs** is a poor fit — it wraps PyTorch's C++ API but loses the
modern Python-side features (Flash Attention, torch.compile, AMP)
that make the training script competitive.

### Practical approaches (ranked by risk)

#### Approach A: Rust orchestrator + Python training (lowest risk)

Port the agent loop to Rust, keep `train.py` as-is. The Rust binary
manages experiments, parses results, handles git operations, and
spawns `uv run train.py` as a subprocess.

- **Effort:** ~1-2 weeks
- **Value:** Type-safe experiment management, better error handling,
  concurrent experiment scheduling, structured logging
- **Risk:** Minimal — Python training is proven
- **Limitation:** Still depends on Python/PyTorch runtime

#### Approach B: Full Rust via Burn (highest ambition)

Port everything to Burn. The model architecture maps cleanly; the
main work is implementing MuonAdamW and validating numerical
equivalence.

- **Effort:** ~4-8 weeks
- **Value:** Single-binary deployment, no Python dependency, runs on
  more GPU vendors via CubeCL, Rust's safety guarantees
- **Risk:** Medium-high — Burn is pre-1.0, you'd be an early adopter
  for serious training workloads. Numerical validation against
  PyTorch is essential
- **Key milestones:**
  1. Port model architecture + AdamW-only training (validate loss curves)
  2. Implement Muon optimizer (validate against Python on small runs)
  3. Port data pipeline (parquet + tokenizer + packing dataloader)
  4. Integrate agent loop
  5. Run head-to-head comparison vs Python version

#### Approach C: Hybrid — Burn model + cudarc for hot paths

Use Burn for the model and autograd, but call Flash Attention and
Muon's orthogonalization kernels via `cudarc` (raw CUDA kernel
launches). Trades portability for performance certainty.

- **Effort:** ~6-10 weeks
- **Value:** Maximum GPU performance, Rust safety for everything
  except kernel code
- **Risk:** High — managing the boundary between Burn tensors and
  raw CUDA buffers is error-prone

### Conclusion

**Yes, the Python code can be converted to Rust.** The ecosystem has
matured enough that all major components have Rust equivalents except
the Muon optimizer (which must be ported manually). Burn is the
recommended framework.

For this project, **Approach B (full Rust via Burn)** aligns best
with the learning goals stated in the project README. Starting with
Approach A as a stepping stone is also reasonable — get the
orchestrator working first, then incrementally port the training.


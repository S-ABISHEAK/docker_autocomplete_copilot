# Work Log

A running record of everything done on this project in collaboration with Claude, top to bottom.
Two tracks of work: (1) evaluating and starting a fine-tuning path for the completion model, and
(2) building and deploying a public-facing showcase page for the project.

---

## 1. Context-window / prompt-engineering review

**Starting question:** would giving the model more surrounding context (more files, more project
info) improve completion quality?

**Findings, from reading the actual pipeline** (`backend/context/collector.py`,
`backend/prompt/builder.py`, `model/architecture.py`, `backend/config.py`):

- The backend only ever collects the current file's prefix (text before the cursor), capped at
  4000 characters — no other files, no project structure, nothing cross-file.
- The model (`ModernTransformer`, `model/architecture.py`) was **trained** with `BLOCK_SIZE = 256`
  — a hard architectural ceiling (RoPE frequency tables are precomputed for exactly that length).
- The **deployed** context window is `CONTEXT_WINDOW_TOKENS = 64` (`backend/config.py`) —
  deliberately far under the 256-token training ceiling, because the model has **no KV cache**: it
  recomputes attention over the entire window on every generated token, so latency scales badly
  with window size (~550ms at 64 tokens vs ~1.7–3.9s at 200 tokens, per the config's own comments).
- Conclusion: the practical next lever isn't "feed it more files" — cross-file context wouldn't
  even fit meaningfully inside a 256-token ceiling. The actual first bottleneck is implementing a
  **KV cache**, which would let the deployed window move toward the trained 256-token ceiling
  without the current latency cliff. Multi-file context is a separate, later change gated on
  retraining with a larger `BLOCK_SIZE`.

---

## 2. Fine-tuning strategy: could a small pretrained model beat the from-scratch one?

**Question:** would fine-tuning an existing small pretrained model (≤1.5B params) on the same
dataset outperform the current 80M from-scratch model?

**Recommendation given:** yes, worth trying — and specifically **Qwen2.5-Coder-0.5B** as the base,
over alternatives considered (Qwen2.5-Coder-1.5B, SmolLM2-360M, TinyLlama-1.1B, CodeGemma/StarCoder2
— the latter two ruled out for having no sub-2B variant). Rationale: Qwen2.5-Coder is already
pretrained on a large code corpus, so fine-tuning it on Dockerfile/YAML/HCL is domain adaptation,
not teaching syntax from zero the way the current from-scratch model had to.

**Honest tradeoffs flagged up front:**
- Breaks the project's current "no quantization tricks needed at this model size" pitch — a
  0.5B–1.5B model needs quantization to stay CPU-viable, and even then per-token latency will be
  materially higher than the 80M model's.
- Training pipeline changes from full pretraining to LoRA/QLoRA fine-tuning (different tooling:
  `peft`, `transformers`, `accelerate`).
- Binary size roughly doubles-to-triples versus the current 409MB ONNX artifact.
- Context window grows hugely "for free" (Qwen2.5-Coder supports far more than 256 tokens).

---

## 3. `finetuned_model/finetune_qwen25_coder_0.5b.ipynb` — the LoRA fine-tuning pipeline

A Kaggle notebook (target: `GPU T4 x2`, ~30GB combined) that LoRA-fine-tunes
`Qwen/Qwen2.5-Coder-0.5B` on **the same corpus and language slice** the original from-scratch model
used: `bigcode/the-stack-dedup`, interleaved `dockerfile` / `yaml` / `hcl`, streamed and tokenized
in the same chunked/resumable-round structure as the original training notebook
(`~/PROJECTS/LLM_FROM_SCRATCH/PIPELINE/docker-model-scratch.ipynb`) — round 0 = fresh LoRA adapter,
every round after = resume adapter + optimizer state, one continuous cosine LR schedule across all
rounds, same `HF_TOKEN`-gated dataset setup.

### Real bugs found and fixed during the build (not just written once and left alone)

1. **Token-id overflow bug, fixed before it ever ran.** The original notebook's pattern packs
   token ids as `uint16`, which just barely fits GPT-2 BPE's ~50k vocab. Qwen's tokenizer has a
   **151,936**-token vocab — `uint16` maxes out at 65,535 and would have silently corrupted every
   id above that. Fixed by using `uint32` throughout.

2. **`notebook_launcher` fork/CUDA crash.** The first multi-GPU attempt used
   `accelerate.notebook_launcher`, which forks worker processes from the Jupyter kernel. On
   Kaggle's image this died with `Cannot re-initialize CUDA in forked subprocess` — something in
   the parent kernel touches CUDA before the fork. **Fix:** rewrote the training loop as a
   standalone `train_worker.py`, generated fresh from the notebook's current config, and launched
   via `accelerate launch` as a genuine separate OS process tree (`torchrun`) instead of a fork —
   sidesteps the whole class of bug rather than chasing the one triggering call.

3. **`torchao` version conflict.** `peft`'s LoRA dispatcher raised
   `ImportError: Found an incompatible version of torchao` against Kaggle's preinstalled build,
   even though this pipeline never uses `torchao` (no quantization here). **Fix:** uninstall it
   (`pip uninstall -y -q torchao`) right after installing the rest of the requirements.

4. **Vocab-driven OOM — the big one.** First real training attempt OOM'd immediately, not from
   the usual "batch too big" activation cost. Qwen's ~152k vocab means the `[batch, seq, vocab]`
   logits tensor dominates memory far more than attention/activation cost does: at the original
   defaults (`BATCH_SIZE=16, BLOCK_SIZE=2048`) that tensor alone needed **~30GB** (fp16 tensor +
   the fp32 upcast the loss function does for numerical stability) — hopeless on a 14.56GB-usable
   T4. **Fix:** dropped to `BATCH_SIZE=4, BLOCK_SIZE=1024` (`~3.7GB` for that tensor),
   `GRAD_ACCUM_STEPS=8` to hold the same effective global batch of 64, and documented the actual
   rule of thumb (`BATCH_SIZE * BLOCK_SIZE` is the ceiling to watch on a large-vocab model, not
   either knob alone).

5. **Eval-survives-but-training-OOMs gap.** Even after the fix above, step-0 eval passed
   (`torch.no_grad()`, activations freed immediately) but the very next real training step OOM'd —
   training must retain every layer's activations for backward, eval doesn't. **Fix:** turned
   **on** `USE_GRADIENT_CHECKPOINTING` by default (it had been off, on the theory of "use more
   VRAM since we have it" — wrong call once the vocab-driven cost was accounted for), plus added
   `PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce fragmentation-driven near-misses.

### Current status

Pipeline is built, debugged through the above issues, and believed runnable end-to-end on Kaggle's
`GPU T4 x2`. **Not yet run to completion / not yet compared against the from-scratch model** —
this is explicitly called out as in-progress on the showcase page (see §4.7 below) rather than
overclaimed as finished.

---

## 4. Public showcase page (`docs/index.html`)

A full single-page HTML/CSS/JS site built to document and showcase the whole
`docker_auto_complete` project — everything the README covers, presented as a designed page rather
than plain markdown, for inclusion in a personal portfolio.

### 4.1 — Initial build

- Single self-contained `.html` file (no build step, no external assets besides Google Fonts and
  one CDN-free inline SVG grain texture).
- **Design system:** "cargo manifest" concept — grounded in the subject (Dockerfiles = shipping
  containers) rather than a generic SaaS template.
  - Typefaces: **Big Shoulders Stencil** (display/stamped headers, crate-stencil lettering),
    **IBM Plex Sans** (body copy), **IBM Plex Mono** (code/data) — deliberately not
    Inter/Space Grotesk.
  - Layout: slim fixed ledger-style side nav that tracks scroll position, long-scroll sections,
    stamped section eyebrows.
- **Hero is a live animated demo, not a video file** — a fake VS Code-style editor window that
  types out a real Dockerfile via JS, pauses, fades in dim ghost-text styled like VS Code's actual
  convention, flashes a "Tab ⇥" chip, accepts it, and loops. Respects `prefers-reduced-motion`
  (shows a static final frame instead of animating).
- **Every number on the page is real**, pulled from the README/config, not invented: 80.8M params,
  the GQA/RoPE/SwiGLU/RMSNorm spec sheet, the 64-vs-256-token context window tradeoff, and the
  actual measured latency figures (44ms/token, 0.7–1.4s typical, the 64-vs-200-token comparison
  rendered as a real bar chart).
- Sections built: hero, What it is (feature grid), How it works (pipeline flow diagram), The model
  (spec table + architecture stack diagram), Performance (stat tiles + latency bar chart), Install
  manifest (the README's 5-step quickstart, with copy buttons and a troubleshooting `<details>`),
  Advanced configuration (env var / extension settings tables), Why local (pull-quote), footer
  (project structure tree, license, GitHub link).

### 4.2 — Standalone deployable file

Wrapped the artifact fragment into a complete `<!DOCTYPE html>` document (`<head>`/`<body>`,
charset, viewport meta) and saved it to **`docs/index.html`** in this repo — chosen specifically
because GitHub Pages supports "deploy from `/docs` on `main`" with no extra config.

### 4.3 — Recolor: black/amber → black/blue (Docker's brand blue)

- Accent changed to `#2496ED` (Docker's actual brand blue), not an arbitrary blue.
- Neutrals re-derived with a cool blue-grey bias (ink/panel/text tokens) instead of the original
  warm-brown bias, so the palette reads as one coherent recomposition rather than a find-replace.
- Ghost-text color, string-literal color, and the "Tab" chip's glow in the animated demo were all
  recolored to match; the green "backend ready" status dot was **left alone** — that's a separate
  semantic signal (healthy/running), not the accent.

### 4.4 — Glassmorphism + ambient 3D backdrop

- Real frosted-glass treatment on UI chrome (nav rail, cards, stat tiles, demo window frame, flow
  steps, badges, the architecture stack panel): translucent backgrounds + `backdrop-filter: blur()`
  + a faint top-edge highlight. Deliberately **not** applied to code blocks, the demo's text area,
  or data tables — glass blur under monospace/tabular content hurts legibility.
- Behind the glass: a fixed ambient layer with two soft radial glows plus a **rotating 3D cluster
  of six glassy panels** arranged in a ring (CSS `rotateY` + `translateZ`, each panel individually
  "billboarded," the whole assembly rotating via one CSS animation — no library).
- **Deliberate choice, called out explicitly:** built as an abstract cluster of container-like
  panels rather than literally reproducing Docker Inc.'s trademarked whale logo in 3D — avoids
  reproducing protected artwork, and arguably fits the subject better anyway (multiple containers,
  not a single mascot).
- Iterated once on visibility: initial blur (46px) was reduced to 20px and opacity raised
  (0.65 → 0.88), with more saturated panel borders/fills, after feedback that it read as "too
  blurry."

### 4.5 — Layout bug fix: dead space on wide screens

`.section`/`footer` were capped at `max-width:1180px` while `.hero` had no cap at all — on any
viewport wider than ~1400px, everything below the hero stopped short and left a visible gap on the
right. Fixed by widening both to `1440px` and adding a 4-column breakpoint for the feature grid
above 1700px viewport width so cards fill the extra room with content instead of stretching thin.

### 4.6 — Deployment

Deployed to Netlify: **https://dockerautocomplete.netlify.app/**. Deployment steps were written up
in detail (GitHub-import method recommended for auto-redeploy on push, vs. drag-and-drop
`docs/index.html` for a one-off).

### 4.7 — More visuals (content-motivated, not decorative)

- **How-it-works pipeline**: each step/arrow briefly lights up in sequence, looping — visualizes a
  request literally traveling through the pipeline the section describes.
- **Performance stat tiles** now count up from 0 to their real value on scroll-into-view.
- **Latency bar chart** grows in from zero width on reveal instead of appearing static.
- **Hero rotor** gained mouse-parallax (subtle drift following the cursor, on top of its existing
  rotation).
- A fifth addition — a scan-line sweeping through the model section's ×20 layer-stack diagram —
  was built, then **removed per feedback** (kept everything else).
- All motion respects `prefers-reduced-motion`.

### 4.8 — More content added

- **New "Engineering decisions" section** (inserted after The Model, before Performance): four
  cards covering the tradeoffs actually made — training from scratch vs. fine-tuning, ONNX Runtime
  vs. PyTorch for serving, the 64-vs-256-token context window decision, and the chunked/resumable
  training approach.
- **Local-vs-cloud comparison table** added into the existing "Why local" section: setup, cost,
  latency, offline capability, and whether code ever leaves the machine — backing up the
  pull-quote with concrete tradeoffs instead of just a philosophy statement.
- **New "What's next" section** (before the footer): an honest, explicitly-not-finished teaser for
  the Qwen2.5-Coder-0.5B LoRA fine-tuning work (§3 above), with a small pulsing "In progress"
  status pill.
- Section lettering in the nav rail and section eyebrows renumbered A→I to fit the two new
  sections in their correct reading order.

---

## Where things stand

- **`docs/index.html`** — live at https://dockerautocomplete.netlify.app/, content and design
  current as of the additions in §4.8. Redeploys automatically on push if Netlify is connected via
  GitHub import; otherwise needs a manual re-drag of the `docs/` folder.
- **`finetuned_model/finetune_qwen25_coder_0.5b.ipynb`** — built and debugged, not yet run to a
  finished, compared result. The showcase page's "What's next" section reflects this honestly.
- Nothing in `backend/`, `model/`, or `extension/` (the actual shipped product) was changed during
  any of this work — all of it was either analysis/planning, a new fine-tuning pipeline living in
  its own `finetuned_model/` folder, or the new `docs/` showcase page.

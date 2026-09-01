# Learning to Follow In-Context Watermark Instructions via Self-Distillation

Official implementation of the in-context watermark instruction following training algorithm presented in the paper:

["Learning to Follow In-Context Watermark Instructions via Self-Distillation"](https://arxiv.org/abs/2608.29030) by Yepeng Liu*, Tianyi Chen*, Xuandong Zhao, Dawn Song, and Yuheng Bu.


## Introduction

In-context watermarking (ICW) prepends an instruction to a query asking the model to embed a statistically detectable signal in its response. It thus equips LLMs with a watermarking interface that third parties can invoke without access to model internals. Its reliability hinges on the LLM following the instruction without degrading answer quality, yet how well current LLMs do so has not been measured. We introduce ICWBench, a benchmark of three verifiable ICW instruction families, each scored on both detectability and answer quality. Evaluating 14 frontier proprietary and open-source LLMs, we find that none of the evaluated LLMs achieves both objectives across all three families. To address this, we propose a self-contained two-stage training method, requiring no distillation from a stronger model, no manual annotation, and no pre-existing ICW IF ability. The first stage, self-distillation with logits perturbation (SDLP), uses the same base LLM as both teacher and student: an instruction-equivalent decoding-time logits perturbation makes the teacher follow the ICW instruction, and the student is trained to match the teacher's output distribution. The second stage applies reinforcement learning with the automatic verifier as the reward. Applied to Qwen3-14B, the weakest of the 14 evaluated LLMs in ICW IF, our method raises average TPR@$1\%$FPR across three ICW instructions from $0.100$ to $0.974$, achieving a more favorable trade-off than the frontier proprietary LLMs we evaluate.

<img width="4789" height="1196" alt="method" src="https://github.com/user-attachments/assets/8ae7b149-ce61-4d85-973f-9ce3f6de12e0" />

This repository contains:

1. **ICWBench** — a benchmark of three verifiable ICW instruction families,
   each paired with a z-statistic verifier.
2. **SDLP** (Self-Distillation with Logits Perturbation) — a frozen copy of the
   base model, given the *clean* query plus an instruction-equivalent
   decoding-time logits perturbation, teaches the ICW-instructed student to
   internalize the instruction (Stage 1).
3. **RL** — GRPO on top of the SDLP checkpoint, with the ICW verifier
   z-statistic as reward (Stage 2).

## ICW instruction families

| Paper term | Code-internal name | Instruction | Verifier |
|---|---|---|---|
| **TSP** — Token-Set Preference | `green` | favor tokens from a green token set `T` | unique-token z-statistic |
| **WIP** — Word-Initial Preference | `initials` | favor words starting with letters in `L` (13 of 26) | unique word-initial z-statistic |
| **SA** — Sentence Acrostic | `acrostics` | sentence-initial letters spell a secret string `S` | LCS z-statistic vs. permutation null |

The code predates the paper terminology; the mapping above applies throughout
(`green` masks are standard watermarking vocabulary). Watermark keys are
per-sample: every training/evaluation query carries its own key.

## Installation

```bash
git clone https://github.com/yepengliu/ICW-IF.git && cd ICW-IF
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -e .            # core library + generation/eval deps (vLLM, torch)
uv pip install -e ./verl       # vendored verl (training)
```

Notes:
- `flash-attn` may need `uv pip install flash-attn --no-build-isolation`.
- IFEval evaluation uses `lighteval`, which **must live in a separate venv**:
  it drags in `antlr4-python3-runtime>=4.11`, which silently breaks
  hydra/omegaconf and hence all verl training. See `evaluation/ifeval.md`.
- Tested with: Python 3.10, torch 2.10 (cu130), vLLM 0.15.x, transformers 4.57,
  ray 2.53. Hardware used in the paper: NVIDIA B200 / RTX PRO 6000 Blackwell.

## Data

All queries derive from [vblagoje/lfqa](https://huggingface.co/datasets/vblagoje/lfqa)
(see `data/README.md`). Two ways to get the training parquets:

**Option A — download the exact paper training data (recommended):**

```bash
hf download JefferyChen453/icw-sdlp-data --repo-type dataset --local-dir data/hf
```

**Option B — resynthesize from scratch** with the logits-perturbation teacher
(`data_gen/README.md` has the full per-task command chains):

```bash
# example: TSP cold-start synthesis (per-sample random keys, delta=3.0)
python data_gen/tsp/run_generate_syn_vllm.py --model_name Qwen/Qwen3-14B ...
```

The synthesized cold-start sets: TSP 3,379 / WIP 865 / SA 2,906 positive
query-response pairs plus 1,000 / 1,000 / 500 clean negatives.

## Stage 1: SDLP

```bash
bash verl/recipe/watermark_kd_ray/scripts/run_sdlp_tsp.sh   # 3 epochs, 8 GPUs
bash verl/recipe/watermark_kd_ray/scripts/run_sdlp_wip.sh   # 2 epochs, 8 GPUs
bash verl/recipe/watermark_kd_ray/scripts/run_sdlp_sa.sh    # 3 epochs, 8 GPUs
```

The teacher is the frozen base model on the clean query with the family's
logits perturbation; the student sees the ICW instruction. Loss: forward KL to
the biased reference plus a clean-reference KL anchor (negatives), computed on
response tokens over the teacher's top-k support. Convert a checkpoint for
inference / Stage 2 with:

```bash
bash tools/convert_ckpt.sh <ckpt>/global_step_N   # -> global_step_N/hf_model
```

## Stage 2: RL

```bash
SDLP_CKPT=<stage1 hf_model> bash verl/recipe/watermark_rl_ray/scripts/run_rl_tsp.sh
SDLP_CKPT=<stage1 hf_model> bash verl/recipe/watermark_rl_ray/scripts/run_rl_wip.sh
SDLP_CKPT=<stage1 hf_model> bash verl/recipe/watermark_rl_ray/scripts/run_rl_sa.sh
```

GRPO with the per-sample verifier z-statistic as reward. The SA reward is
protected by anti-reward-hack regex gates (markdown bold, orphan letters,
letter headings, secret dumps) that zero the reward before the detector runs.

## Evaluation (ICWBench)

Evaluate **any local HF checkpoint** on the three families — generation with
per-query watermark keys, detection, and AUC / TPR@1%FPR / TPR@10%FPR
aggregation:

```bash
bash evaluation/icwbench/run_icwbench.sh --model <path-or-hf-id>
```

Every one of the 477 held-out queries carries its own independently sampled
key, disjoint from all training keys (`data/eval/`, see `data/README.md`).
Note: the paper's reported numbers were produced under a shared-key protocol;
the released per-query-key protocol is stricter, so freshly evaluated numbers
can differ slightly from the paper tables.

Additional axes (see `evaluation/README.md`):

- **Perplexity** — `evaluation/quality/compute_ppl.py` (Qwen3-235B judge model)
- **LLM judge** — `evaluation/quality/llm_judge.py` (fluency / watermark
  invisibility, requires `OPENAI_API_KEY`)
- **IFEval** — `evaluation/ifeval.md` (general instruction-following retention)
- **Robustness** — `evaluation/robustness/` (paraphrase, 30% word replacement,
  30% word deletion)

## Repository layout

```
watermark/     core library: masks, perturbations, detectors, ICW prompts, vLLM hooks
data/          LFQA splits + per-query-key evaluation manifests + tokenizer stats
data_gen/      Stage-0 cold-start data synthesis (three families)
verl/          vendored verl fork; recipes: watermark_kd_ray (SDLP), watermark_rl_ray (RL)
evaluation/    ICWBench + quality + robustness evaluation
tools/         checkpoint conversion
```

## Citation

```bibtex
@article{liu2026icwif,
  title={Learning to follow in-context watermark instructions via self-distillation},
  author={Liu, Yepeng and Chen, Tianyi and Zhao, Xuandong and Song, Dawn and Bu, Yuheng},
  journal={arXiv preprint arXiv:2608.29030},
  year={2026}
}
```

## Acknowledgements

Training is built on [verl](https://github.com/volcengine/verl) (vendored under
`verl/` with our watermark recipes and minor core fixes). Queries come from the
[vblagoje/lfqa](https://huggingface.co/datasets/vblagoje/lfqa) dataset.

## License

Apache 2.0 (including the vendored verl fork).

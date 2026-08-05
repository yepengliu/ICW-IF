# Training recipes

Two verl-based recipes implement the two-stage training pipeline of the paper:

## `watermark_kd_ray/` — Stage 1: SDLP (Self-Distillation with Logits Perturbation)

Offline distillation. A frozen copy of the base model, given the **clean**
query and a decoding-time logits perturbation, is the teacher; the student
sees the **ICW instruction + query** and is trained to match the teacher's
(top-k truncated) next-token distribution on response tokens. Negative rows
anchor the student to the clean reference distribution.

Launchers (canonical paper hyperparameters baked in):

```bash
bash recipe/watermark_kd_ray/scripts/run_sdlp_tsp.sh   # Token-Set Preference
bash recipe/watermark_kd_ray/scripts/run_sdlp_wip.sh   # Word-Initial Preference
bash recipe/watermark_kd_ray/scripts/run_sdlp_sa.sh    # Sentence Acrostic
```

## `watermark_rl_ray/` — Stage 2: RL (GRPO)

GRPO on top of the SDLP checkpoint; the reward is the per-sample ICW verifier
z-statistic (LCS-based for SA, unique-token z for TSP/WIP). The SA reward is
protected by anti-reward-hack regex gates (`md_bold`, `orphan_letter`,
`letter_heading`, `secret_dump`) that zero the reward before the detector runs.

```bash
SDLP_CKPT=<stage1 hf_model dir> bash recipe/watermark_rl_ray/scripts/run_rl_tsp.sh
SDLP_CKPT=<stage1 hf_model dir> bash recipe/watermark_rl_ray/scripts/run_rl_wip.sh
SDLP_CKPT=<stage1 hf_model dir> bash recipe/watermark_rl_ray/scripts/run_rl_sa.sh
```

Checkpoints are saved under `verl/checkpoints/<project>/<experiment>/`; convert
FSDP shards to a HuggingFace model with `verl/scripts/legacy_model_merger.py`
(see the repository README).

Both recipes assume the training parquets from the
[icw-sdlp-data](https://huggingface.co/datasets/JefferyChen453/icw-sdlp-data)
dataset have been downloaded to `data/hf/` (see the repository README).

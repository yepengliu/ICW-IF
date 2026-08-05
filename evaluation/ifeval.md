# IFEval (general instruction-following retention)

We evaluate general instruction-following with
[lighteval](https://github.com/huggingface/lighteval)'s `ifeval` task.

## ⚠️ Use a dedicated virtual environment

Do **not** install lighteval into the training environment. lighteval pulls in
`antlr4-python3-runtime>=4.11`, while the training stack (hydra / omegaconf used
by verl) requires `antlr4-python3-runtime==4.9.3`. Installing lighteval into the
training venv silently breaks every training entry point (the failure surfaces
as an error inside `import hydra`).

```bash
python -m venv ~/venvs/lighteval
source ~/venvs/lighteval/bin/activate
pip install lighteval[vllm]
```

## Run

```bash
lighteval vllm \
    "model_name=<CKPT_PATH_OR_HF_ID>,dtype=bfloat16,gpu_memory_utilization=0.9" \
    "ifeval|0" \
    --output-dir outputs/ifeval \
    --save-details
```

- `<CKPT_PATH_OR_HF_ID>` — e.g. `Qwen/Qwen3-14B` (baseline) or a converted
  HF checkpoint directory produced by the training pipeline.
- Results land in `outputs/ifeval/results/<model>/results_<timestamp>.json`;
  report `prompt_level_strict_acc` and `inst_level_strict_acc` alongside the
  loose variants.
- Compare each trained checkpoint against its base model evaluated with the
  identical command; the release claim is *retention* (no significant drop),
  not absolute score.

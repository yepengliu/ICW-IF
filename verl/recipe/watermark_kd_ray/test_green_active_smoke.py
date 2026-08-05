#!/usr/bin/env python3
"""Smoke test for green active-mask KD loss (green_active_mode none/mask/cleanref).
CPU-only, no distributed. Validates: finite loss, backward works, active counts
sane, mask vs cleanref bucket behavior, and that active-mask differs from full-KL.
Run: cd verl && python -m recipe.watermark_kd_ray.test_green_active_smoke
"""
import torch
import torch.nn.functional as F
from recipe.watermark_kd_ray.loss import compute_watermark_kd_loss

torch.manual_seed(0)
V = 60
NS = 3                      # 2 green pos + 1 neg
TOK = 10                    # response tokens per sample
device = "cpu"

# green masks (NS, V): ~25% green for pos; neg sample mask irrelevant (fraction 0)
green_masks = torch.zeros(NS, V, dtype=torch.bool)
for i in range(NS):
    idx = torch.randperm(V)[: int(0.25 * V)]
    green_masks[i, idx] = True

sample_index = torch.arange(NS).unsqueeze(1).expand(-1, TOK).reshape(-1)   # (NS*TOK,)
actor_logits = torch.randn(NS * TOK, V, requires_grad=True)
ref_logits = torch.randn(NS * TOK, V)
input_ids_rolled = torch.randint(0, V, (NS * TOK,))
sample_is_negative = torch.tensor([False, False, True])
sample_task_ids = torch.tensor([0, 0, 2])   # 0=green, 2=neg

def green_counts():
    """Replicate worker count: target-is-green among green-pos tokens."""
    n_act = n_non = 0
    for j in range(NS * TOK):
        i = int(sample_index[j])
        if sample_is_negative[i]:
            continue
        if bool(green_masks[i, input_ids_rolled[j]]):
            n_act += 1
        else:
            n_non += 1
    return float(n_act), float(n_non)

n_green_active, n_green_nonactive = green_counts()
n_pos = float((~sample_is_negative[sample_index]).sum())      # green pos tokens
n_neg = float((sample_is_negative[sample_index]).sum())
print(f"counts: green_active={n_green_active} green_nonactive={n_green_nonactive} "
      f"(active_frac={n_green_active/(n_green_active+n_green_nonactive):.2f}) n_pos={n_pos} n_neg={n_neg}")

common = dict(
    ref_logits=ref_logits, input_ids_rolled=input_ids_rolled, sample_index=sample_index,
    green_masks=green_masks, strength=5.0,
    ce_loss_weight=0.0, green_loss_weight=0.0,
    kl_biased_ref_actor_weight=1.0, reverse_kl_biased_ref_actor_weight=0.0,
    kl_ref_actor_weight=1.0, reverse_kl_ref_actor_weight=0.0,
    kl_biased_actor_actor_weight=0.0, reverse_kl_biased_actor_actor_weight=0.0,
    batch_num_tokens=float(NS * TOK), dp_size=1,
    sample_is_negative=sample_is_negative, sample_task_ids=sample_task_ids,
    loss_normalization_mode="per_task",
    batch_num_pos_nonacr_tokens=n_pos, batch_num_neg_tokens=n_neg,
    batch_num_green_active_tokens=n_green_active,
    batch_num_green_nonactive_tokens=n_green_nonactive,
)

results = {}
for mode in ["none", "mask", "cleanref"]:
    al = actor_logits.detach().clone().requires_grad_(True)
    loss, m = compute_watermark_kd_loss(actor_logits=al, green_active_mode=mode, **common)
    loss.backward()
    gnorm = al.grad.norm().item()
    results[mode] = (loss.item(), m, gnorm)
    print(f"\n[mode={mode}] loss={loss.item():.5f} grad_norm={gnorm:.5f} finite={torch.isfinite(loss).item()}")
    print(f"   metrics: " + ", ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in m.items() if k in
        ("kl_biased_ref_actor","kl_ref_actor","kl_ref_pos","n_green_active_tokens","green_active_frac","total_loss")))

# ---- Assertions ----
ok = True
for mode in ["none","mask","cleanref"]:
    l, m, g = results[mode]
    assert torch.isfinite(torch.tensor(l)), f"{mode} loss not finite"
    assert g > 0, f"{mode} grad_norm should be >0 (got {g})"
# none = full-KL (biased on all pos tokens); mask/cleanref = biased on active only → different loss
assert abs(results["none"][0] - results["mask"][0]) > 1e-4, "mask should differ from full-KL"
# cleanref adds a pos clean-ref term that mask lacks → cleanref has kl_ref_pos metric, mask does not
assert "kl_ref_pos" in results["cleanref"][1], "cleanref must emit kl_ref_pos"
assert "kl_ref_pos" not in results["mask"][1], "mask must NOT emit kl_ref_pos"
# green_active_frac present in mask/cleanref, absent in none
assert "green_active_frac" in results["mask"][1] and "green_active_frac" not in results["none"][1]
# cleanref total loss should exceed mask (extra non-neg clean-ref term, all positive KL)
assert results["cleanref"][0] > results["mask"][0] - 1e-6, "cleanref should add a non-negative pos clean-ref term"
print("\nALL SMOKE ASSERTIONS PASSED ✅")

# ---- repeat with english_vocab_mask (only_english path) ----
emask = torch.zeros(V, dtype=torch.bool); emask[: int(0.7*V)] = True
for mode in ["mask","cleanref"]:
    al = actor_logits.detach().clone().requires_grad_(True)
    loss, m = compute_watermark_kd_loss(actor_logits=al, green_active_mode=mode,
                                        english_vocab_mask=emask, **common)
    loss.backward()
    assert torch.isfinite(loss) and al.grad.norm().item() > 0, f"english-mask {mode} failed"
    print(f"[english_vocab_mask mode={mode}] loss={loss.item():.5f} OK")
print("ENGLISH-MASK PATH OK ✅")

# ---- top-K on biased green-active teacher (distill_topk_biased_ref) ----
# Verifies the green-active biased branch honors top-K (mirrors initials/acrostic),
# so "cleanref + topk" actually stacks both levers instead of silently dropping K.
K = 10   # < |english|=42 so it truncates
for mode in ["mask", "cleanref"]:
    al = actor_logits.detach().clone().requires_grad_(True)
    loss_k, mk = compute_watermark_kd_loss(actor_logits=al, green_active_mode=mode,
                                           english_vocab_mask=emask,
                                           distill_topk_biased_ref=K, **common)
    loss_k.backward()
    assert torch.isfinite(loss_k) and al.grad.norm().item() > 0, f"topk {mode} failed"
    al2 = actor_logits.detach().clone().requires_grad_(True)
    loss_full, _ = compute_watermark_kd_loss(actor_logits=al2, green_active_mode=mode,
                                             english_vocab_mask=emask, **common)
    assert abs(loss_k.item() - loss_full.item()) > 1e-5, \
        f"topk {mode} should differ from full-english (K not applied?)"
    print(f"[topK={K} mode={mode}] loss={loss_k.item():.5f} (full-english={loss_full.item():.5f}) OK")
print("TOP-K GREEN-ACTIVE PATH OK ✅")

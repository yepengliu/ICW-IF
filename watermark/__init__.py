"""Core library for ICW-SDLP.

Modules
-------
- ``gptwm``               seed-keyed green-list construction, logit-bias warper,
                          z-score detector (TSP)
- ``gptwm_initials``      word-initial letter-set masks and detector (WIP)
- ``gptwm_acrostics``     sentence-acrostic verifier wrapper (SA)
- ``gptwm_acrostics_bias``stateful acrostic decoding-time bias controller (SA teacher)
- ``acrostics_icw``       acrostic ICW prompts, markdown-aware extractor, LCS matching
- ``acrostics_zstat``     permutation-null z-statistic for SA
- ``gptwm_incontext``     green-list -> in-context prompt string serialisation
- ``gptwm_vllm_config``   per-request watermark logits processors for vLLM
- ``prompt``              ICW instruction template registry
- ``dataset``             JSONL loading and chat-template formatting helpers
"""

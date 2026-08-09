# Learning to Follow In-Context Watermark Instructions via Self-Distillation

Official implementation of the in-context watermark instruction following training algorithm presented in the paper:

["Learning to Follow In-Context Watermark Instructions via Self-Distillation"]() by Yepeng Liu*, Tianyi Chen*, Xuandong Zhao, Dawn Song, and Yuheng Bu.


## Introduction

In-context watermarking (ICW) prepends an instruction to a query asking the model to embed a statistically detectable signal in its response. It thus equips LLMs with a watermarking interface that third parties can invoke without access to model internals. Its reliability hinges on the LLM following the instruction without degrading answer quality, yet how well current LLMs do so has not been measured. We introduce ICWBench, a benchmark of three verifiable ICW instruction families, each scored on both detectability and answer quality. Evaluating 14 frontier proprietary and open-source LLMs, we find that none of the evaluated LLMs achieves both objectives across all three families. To address this, we propose a self-contained two-stage training method, requiring no distillation from a stronger model, no manual annotation, and no pre-existing ICW IF ability. The first stage, self-distillation with logits perturbation (SDLP), uses the same base LLM as both teacher and student: an instruction-equivalent decoding-time logits perturbation makes the teacher follow the ICW instruction, and the student is trained to match the teacher's output distribution. The second stage applies reinforcement learning with the automatic verifier as the reward. Applied to Qwen3-14B, the weakest of the 14 evaluated LLMs in ICW IF, our method raises average TPR@$1\%$FPR across three ICW instructions from $0.100$ to $0.974$, achieving a more favorable trade-off than the frontier proprietary LLMs we evaluate.

<img width="4789" height="1196" alt="method" src="https://github.com/user-attachments/assets/8ae7b149-ce61-4d85-973f-9ce3f6de12e0" />


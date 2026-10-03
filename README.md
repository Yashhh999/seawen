# seawen

QLoRA fine-tune of `Qwen/Qwen3.5-9B` for Kaggle's 2× T4.

Targets, in one run:

- coding (execution-filtered public instructions)
- reasoning (math chain-of-thought with the final result kept)
- abstention (say "I don't know" instead of inventing a fact)
- fewer refusals on blunt, legal prompts (politics, fiction with ordinary swearing, direct opinions)

The shipped direct-answer file does **not** contain sexual content, crime instructions, or self-harm. Add your own legal rows to [data/direct_answers.jsonl](data/direct_answers.jsonl) if you want a wider surface. A measured 0% refusal rate is not the goal of this mix: pushing that far is what drops calibration.

## What this will not do

It will not raise AA-Briefcase, Omniscience accuracy, or a general intelligence index. Abstention can improve an index that penalizes guessing. Coding and reasoning can move a bit. Two T4s cannot do bf16 LoRA of the 9B, and Unsloth only trains on one GPU, so the second T4 sits idle. 4-bit QLoRA is a VRAM compromise Unsloth does not recommend for Qwen3.5; it is the only option that fits.

Default sequence length is **2048**, not 4096. A 9B 4-bit run at 4096 OOMs on a 16GB T4 often enough that "just run it" would fail. Set `SEAWEN_MAX_SEQ=4096` only after a short run survives.

## Kaggle

1. New Notebook. Settings: **GPU T4 x2**, **Internet On**.
2. Paste the cells from [kaggle/seawen_train.py](kaggle/seawen_train.py), or upload this repo and run them.
3. When it finishes, the adapter is in `/kaggle/working/seawen-adapter`. Download that folder from the notebook output.

First launch compiles Qwen3.5's Mamba kernels and looks hung. That is one-time.

```bash
# defaults: 4000 samples, seq 2048, 1 epoch, lr 1e-4, r=16, alpha=32
python -m seawen.train
python -m seawen.eval_smoke --adapter /kaggle/working/seawen-adapter
```

Smoke eval is eight prompts. A guess on an unknowable question is a failure. A refusal on a legal direct prompt is a failure. It is not Terminal-Bench.

## Mix

Remote rows fill whatever budget the local files do not use. Local rows are repeated at most 4 times so two dozen refusal demos cannot overwrite coding.

| slice | source |
|---|---|
| coding | `bigcode/self-oss-instruct-sc2-exec-filter-50k`, else Magicoder-OSS-Instruct |
| reasoning | `AI-MO/NuminaMath-CoT`, else OpenR1-Math-220k |
| abstention | [data/abstention.jsonl](data/abstention.jsonl), plus a few easy answers so it does not refuse everything |
| direct | [data/direct_answers.jsonl](data/direct_answers.jsonl) |

Loss is masked to the assistant span when the chat template markers can be detected.

## Knobs

| env | default |
|---|---|
| `SEAWEN_MAX_SAMPLES` | 4000 |
| `SEAWEN_MAX_SEQ` | 2048 |
| `SEAWEN_EPOCHS` | 1 |
| `SEAWEN_LR` | 1e-4 |
| `SEAWEN_RANK` / `SEAWEN_ALPHA` | 16 / 32 |
| `SEAWEN_OUTPUT` | `/kaggle/working/seawen-adapter` |
| `SEAWEN_GPU` | 0 |

`python -m seawen.train --dry-run` builds the mix and exits before loading the model.

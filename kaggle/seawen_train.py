# Run this file's cells in a Kaggle notebook.
# Settings: Accelerator = GPU T4 x2, Internet = On.
# The second T4 is unused. Unsloth trains on one GPU, and a 9B bf16 LoRA does not fit.

# %%
import os
import subprocess
import sys

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "unsloth",
        "trl",
        "datasets",
        "peft",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub",
    ]
)

repo = "/kaggle/working/seawen"
if not os.path.exists(os.path.join(repo, "seawen", "train.py")):
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/Yashhh999/seawen.git", repo])
os.chdir(repo)
sys.path.insert(0, repo)

# 4000 samples is a few hours, not a full 12h quota. Raise SEAWEN_MAX_SAMPLES if the first run looks healthy.
os.environ.setdefault("SEAWEN_MAX_SAMPLES", "4000")
os.environ.setdefault("SEAWEN_MAX_SEQ", "2048")
os.environ.setdefault("SEAWEN_OUTPUT", "/kaggle/working/seawen-adapter")

from seawen.train import main

main()

# %%
# Smoke test after training. Read the prints. Do not treat this as a benchmark score.
from seawen.eval_smoke import main as eval_main

eval_main()

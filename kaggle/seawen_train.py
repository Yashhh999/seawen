# One cell. Kaggle settings: GPU T4 x2, Internet On.
import os
import shutil
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
url = "https://github.com/Yashhh999/seawen.git"
if os.path.isdir(os.path.join(repo, ".git")):
    subprocess.check_call(["git", "-C", repo, "pull", "--ff-only"])
else:
    if os.path.exists(repo):
        shutil.rmtree(repo)
    subprocess.check_call(["git", "clone", "--depth", "1", url, repo])

os.chdir(repo)
sys.path.insert(0, repo)
for name in list(sys.modules):
    if name == "seawen" or name.startswith("seawen."):
        del sys.modules[name]

os.environ["SEAWEN_MAX_SAMPLES"] = "4000"
os.environ["SEAWEN_MAX_SEQ"] = "2048"
os.environ["SEAWEN_SAVE_STEPS"] = "5"
os.environ["SEAWEN_HUB_ID"] = "Yashhh999/seawen"
os.environ["SEAWEN_OUTPUT"] = "/kaggle/working/seawen-adapter"

from seawen.train import main

main()

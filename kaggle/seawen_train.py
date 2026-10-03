# One cell. Kaggle settings: GPU T4 x2, Internet On. Secret label: HF_TOKEN.
import os
import shutil
import subprocess
import sys
from pathlib import Path

def _kaggle_token():
    names = ("HF_TOKEN", "HF_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "hf_token")
    for key, value in os.environ.items():
        if value and value.startswith("hf_") and ("HF" in key.upper() or "HUGG" in key.upper()):
            return value.strip()
    for name in names:
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    from kaggle_secrets import UserSecretsClient
    client = UserSecretsClient()
    for name in names:
        try:
            value = client.get_secret(name)
        except Exception:
            continue
        if value and str(value).strip():
            print(f"using Kaggle secret {name}")
            return str(value).strip()
    raise SystemExit("No Hugging Face token. Add-ons -> Secrets, label HF_TOKEN, attach it, run again.")

token = _kaggle_token()
os.environ["HF_TOKEN"] = token
os.environ["HUGGING_FACE_HUB_TOKEN"] = token
os.environ["HUGGINGFACE_HUB_TOKEN"] = token
token_path = Path.home() / ".cache" / "huggingface" / "token"
token_path.parent.mkdir(parents=True, exist_ok=True)
token_path.write_text(token)

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

import unsloth  # before huggingface_hub, which pulls transformers
from huggingface_hub import login
login(token=token, add_to_git_credential=False)
print(f"HF auth ok ({token[:3]}…{token[-4:]})")

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

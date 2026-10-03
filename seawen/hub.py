"""Push trainer checkpoints to Yashhh999/seawen and pull the latest complete one back.

A checkpoint counts only if it has trainer_state.json and the LoRA weights.
latest.json is written after the upload, so a crash mid-push cannot make the
next run resume a half-written folder.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

HUB_ID = os.environ.get("SEAWEN_HUB_ID", "Yashhh999/seawen")
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_model.bin")
SECRET_NAMES = (
    "HF_HUB_TOKEN",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "hf_token",
)


def get_token() -> str | None:
    for key, value in os.environ.items():
        if not value or not value.startswith("hf_"):
            continue
        upper = key.upper()
        if "HF" in upper or "HUGG" in upper or upper in {"TOKEN", "HUGGINGFACE"}:
            return value.strip()
    for key in SECRET_NAMES:
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
    except Exception:
        return None
    for name in SECRET_NAMES:
        try:
            value = client.get_secret(name)
        except Exception:
            continue
        if value and str(value).strip():
            print(f"using Kaggle secret {name}")
            return str(value).strip()
    return None


def activate_token(token: str) -> None:
    """Set every place Hugging Face looks, before the first download."""
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    path = Path.home() / ".cache" / "huggingface" / "token"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token)
    os.chmod(path, 0o600)
    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)
    print(f"HF auth ok ({token[:3]}…{token[-4:]})")


def _step(name: str) -> int | None:
    prefix = "checkpoint-"
    leaf = name.split("/")[0]
    if not leaf.startswith(prefix):
        return None
    try:
        return int(leaf[len(prefix) :])
    except ValueError:
        return None


def _complete(names: set[str]) -> bool:
    return "trainer_state.json" in names and any(name in names for name in ADAPTER_FILES)


def local_checkpoints(output_dir: str) -> list[tuple[int, str]]:
    root = Path(output_dir)
    found = []
    if not root.is_dir():
        return found
    for path in root.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        step = _step(path.name)
        if step is None:
            continue
        names = {child.name for child in path.iterdir()}
        if _complete(names):
            found.append((step, str(path)))
        else:
            print(f"ignoring incomplete local {path.name}")
    return sorted(found)


def _api(token: str):
    from huggingface_hub import HfApi

    return HfApi(token=token)


def ensure_repo(token: str, repo_id: str = HUB_ID) -> None:
    from huggingface_hub import create_repo

    create_repo(repo_id, repo_type="model", exist_ok=True, private=False, token=token)


def remote_checkpoints(token: str, repo_id: str = HUB_ID) -> list[int]:
    api = _api(token)
    try:
        files = api.list_repo_files(repo_id, repo_type="model")
    except Exception as exc:
        text = str(exc).lower()
        if "404" in text or "not found" in text:
            print(f"{repo_id} does not exist yet")
            ensure_repo(token, repo_id)
            return []
        raise
    grouped: dict[int, set[str]] = {}
    for path in files:
        step = _step(path)
        if step is None:
            continue
        grouped.setdefault(step, set()).add(path.split("/")[-1])
    good = []
    for step, names in grouped.items():
        if _complete(names):
            good.append(step)
        else:
            print(f"ignoring incomplete hub checkpoint-{step}")
    return sorted(good)


def pull_checkpoint(token: str, step: int, output_dir: str, repo_id: str = HUB_ID) -> str | None:
    from huggingface_hub import snapshot_download

    os.makedirs(output_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"checkpoint-{step}/*"],
        local_dir=output_dir,
        token=token,
    )
    path = os.path.join(output_dir, f"checkpoint-{step}")
    if os.path.isdir(path) and _complete(set(os.listdir(path))):
        return path
    print(f"downloaded checkpoint-{step} but it is still incomplete")
    return None


def pull_mix(token: str, output_dir: str, repo_id: str = HUB_ID) -> str | None:
    from huggingface_hub import hf_hub_download

    os.makedirs(output_dir, exist_ok=True)
    dest = os.path.join(output_dir, "mix.jsonl")
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename="mix.jsonl",
            repo_type="model",
            local_dir=output_dir,
            token=token,
        )
    except Exception as exc:
        print(f"no mix.jsonl on the hub ({exc})")
        return None


def push_mix(token: str, path: str, repo_id: str = HUB_ID) -> None:
    api = _api(token)
    ensure_repo(token, repo_id)
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo="mix.jsonl",
        repo_id=repo_id,
        repo_type="model",
        commit_message="training mix",
        token=token,
    )
    print("pushed mix.jsonl")


def push_checkpoint(token: str, folder: str, step: int, repo_id: str = HUB_ID, keep: int = 2) -> None:
    api = _api(token)
    ensure_repo(token, repo_id)
    api.upload_folder(
        folder_path=folder,
        path_in_repo=f"checkpoint-{step}",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"checkpoint {step}",
        token=token,
    )
    marker = {"step": step, "folder": f"checkpoint-{step}"}
    api.upload_file(
        path_or_fileobj=json.dumps(marker).encode(),
        path_in_repo="latest.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"latest checkpoint {step}",
        token=token,
    )
    print(f"pushed checkpoint-{step}")
    try:
        complete = remote_checkpoints(token, repo_id)
    except Exception as exc:
        print(f"could not list checkpoints to prune ({exc})")
        return
    for old in complete[:-keep]:
        try:
            api.delete_folder(
                f"checkpoint-{old}",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"drop checkpoint {old}",
                token=token,
            )
            print(f"dropped hub checkpoint-{old}")
        except Exception as exc:
            print(f"could not drop checkpoint-{old} ({exc})")


def push_adapter(token: str, folder: str, repo_id: str = HUB_ID) -> None:
    api = _api(token)
    ensure_repo(token, repo_id)
    api.upload_folder(
        folder_path=folder,
        repo_id=repo_id,
        repo_type="model",
        commit_message="final adapter",
        allow_patterns=[
            "adapter_*",
            "tokenizer*",
            "special_tokens*",
            "chat_template*",
            "added_tokens*",
            "merges.txt",
            "vocab.json",
            "*.model",
        ],
        ignore_patterns=["checkpoint-*/*", "mix.jsonl"],
        token=token,
    )
    print(f"pushed adapter to {repo_id}")


def resolve_resume(output_dir: str, token: str | None, repo_id: str = HUB_ID) -> str | None:
    local = local_checkpoints(output_dir)
    local_step = local[-1][0] if local else -1
    remote_step = -1
    if token:
        try:
            remote = remote_checkpoints(token, repo_id)
            remote_step = remote[-1] if remote else -1
        except Exception as exc:
            print(f"hub lookup failed, using local state only ({exc})")
    if remote_step > local_step and token:
        print(f"hub checkpoint-{remote_step} is newer than local {local_step}")
        try:
            pulled = pull_checkpoint(token, remote_step, output_dir, repo_id)
        except Exception as exc:
            print(f"download of checkpoint-{remote_step} failed ({exc})")
            pulled = None
        if pulled:
            return pulled
        print("falling back to the local checkpoint" if local_step >= 0 else "starting fresh")
    if local_step >= 0:
        print(f"resuming local checkpoint-{local_step}")
        return local[-1][1]
    print("no checkpoint found, starting at step 0")
    return None

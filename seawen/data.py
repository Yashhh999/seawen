"""Build the 40/35/15/10 mix. Coding and reasoning come from public datasets.

Abstention and direct-answer rows ship in this repo. Direct answers are blunt
replies to over-refused but legal prompts. They are not exploit, self-harm, or
sexual-content targets.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

MIX = {
    "coding": 0.40,
    "reasoning": 0.35,
    "abstention": 0.15,
    "direct": 0.10,
}

CODING_SOURCES = (
    ("bigcode/self-oss-instruct-sc2-exec-filter-50k", ("instruction", "problem"), ("response", "solution")),
    ("ise-uiuc/Magicoder-OSS-Instruct-75K", ("problem", "instruction"), ("solution", "response")),
)
REASONING_SOURCES = (
    ("AI-MO/NuminaMath-CoT", ("problem", "question"), ("solution", "answer")),
    ("open-r1/OpenR1-Math-220k", ("problem", "question"), ("solution", "answer")),
)

REFUSAL_PREFIXES = (
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't assist",
    "i can’t assist",
    "as an ai",
    "i must refuse",
)


def _counts(n: int) -> dict[str, int]:
    raw = {k: int(n * v) for k, v in MIX.items()}
    raw["direct"] += n - sum(raw.values())
    return raw


def _first(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _boxed(solution: str) -> tuple[str, str]:
    marker = "\\boxed{"
    start = solution.rfind(marker)
    if start < 0:
        text = solution.strip()
        return text, ""
    depth = 0
    for i, ch in enumerate(solution[start + len(marker) - 1 :], start + len(marker) - 1):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                inside = solution[start + len(marker) : i].strip()
                think = solution[:start].strip()
                return think or solution.strip(), inside
    return solution.strip(), ""


def as_message(user: str, assistant: str, think: str | None = None) -> dict:
    body = assistant.strip()
    if think and "<think>" not in body:
        body = f"<think>\n{think.strip()}\n</think>\n{body}"
    elif "<think>" not in body:
        body = f"<think>\nThe user wants a direct, accurate answer. I should not pad it or invent facts.\n</think>\n{body}"
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": body},
        ]
    }


def _is_refusal(text: str) -> bool:
    head = text.strip().lower()[:80]
    return any(head.startswith(p) for p in REFUSAL_PREFIXES)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def local_rows(kind: str) -> list[dict]:
    if kind == "direct":
        path = DATA_DIR / "direct_answers.jsonl"
        return [as_message(row["user"], row["assistant"]) for row in _load_jsonl(path)]
    path = DATA_DIR / "abstention.jsonl"
    out = []
    for row in _load_jsonl(path):
        if row.get("kind") == "answer":
            out.append(as_message(row["user"], row["assistant"], think="This one I can answer directly."))
        else:
            out.append(
                as_message(
                    row["user"],
                    row["assistant"],
                    think="I do not have a reliable source for this. Guessing would be a hallucination.",
                )
            )
    return out


def _stream_pairs(sources, limit: int, seed: int) -> list[tuple[str, str]]:
    from datasets import load_dataset

    errors = []
    for name, user_keys, answer_keys in sources:
        try:
            dataset = load_dataset(name, split="train", streaming=True)
            picked = []
            for index, row in enumerate(dataset):
                if index > limit * 40:
                    break
                user = _first(row, user_keys)
                answer = _first(row, answer_keys)
                if not user or not answer or _is_refusal(answer):
                    continue
                if len(user) < 20 or len(answer) < 20:
                    continue
                picked.append((user, answer))
                if len(picked) >= limit * 3:
                    break
            if len(picked) < min(32, limit):
                errors.append(f"{name}: only {len(picked)} usable rows")
                continue
            random.Random(seed).shuffle(picked)
            return picked[: limit * 3]
        except Exception as exc:  # dataset missing, gated, or offline
            errors.append(f"{name}: {exc}")
    raise RuntimeError("No remote dataset loaded. " + " | ".join(errors))


def remote_rows(kind: str, limit: int, seed: int) -> list[dict]:
    sources = CODING_SOURCES if kind == "coding" else REASONING_SOURCES
    pairs = _stream_pairs(sources, limit, seed)
    rows = []
    for user, answer in pairs[:limit]:
        if kind == "reasoning":
            think, boxed = _boxed(answer)
            assistant = f"\\boxed{{{boxed}}}" if boxed else answer
            think = think or "Solve it carefully, then give the result."
            rows.append(as_message(user, assistant, think=think))
        else:
            rows.append(
                as_message(
                    user,
                    answer,
                    think="I'll write the solution so it matches the request and would pass a straightforward test.",
                )
            )
    return rows


def _take(rows: list[dict], n: int, seed: int, max_repeat: int = 4) -> list[dict]:
    if not rows or n <= 0:
        return []
    cap = len(rows) * max_repeat
    n = min(n, cap)
    rng = random.Random(seed)
    pool = rows * max_repeat
    return rng.sample(pool, n)


def build_mix(max_samples: int, seed: int = 3407) -> list[dict]:
    counts = _counts(max_samples)
    direct_all = local_rows("direct")
    abstain_all = local_rows("abstention")
    direct = _take(direct_all, counts["direct"], seed)
    abstention = _take(abstain_all, counts["abstention"], seed + 1)
    # Unused direct/abstention budget goes back to coding and reasoning.
    # Repeating two dozen refusal rows hundreds of times is how coding degrades.
    leftover = max_samples - (len(direct) + len(abstention))
    coding_n = int(leftover * MIX["coding"] / (MIX["coding"] + MIX["reasoning"]))
    reasoning_n = leftover - coding_n
    coding = remote_rows("coding", coding_n, seed + 2)
    reasoning = remote_rows("reasoning", reasoning_n, seed + 3)
    if len(coding) < coding_n or len(reasoning) < reasoning_n:
        raise RuntimeError(
            f"Short remote mix: coding {len(coding)}/{coding_n}, "
            f"reasoning {len(reasoning)}/{reasoning_n}"
        )
    mixed = coding + reasoning + abstention + direct
    random.Random(seed).shuffle(mixed)
    print(
        f"counts coding={len(coding)} reasoning={len(reasoning)} "
        f"abstention={len(abstention)} direct={len(direct)}"
    )
    return mixed


def render_text(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

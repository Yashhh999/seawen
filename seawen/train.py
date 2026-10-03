"""QLoRA SFT for Qwen3.5-9B on one Kaggle T4.

4-bit is a VRAM compromise. Unsloth advises against QLoRA on Qwen3.5 because
the quantization error is higher than usual. Two T4s cannot hold a bf16 LoRA
at a useful sequence length, and Unsloth trains on one GPU, so the second T4
stays idle on purpose.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("SEAWEN_GPU", "0"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def parse_args():
    parser = argparse.ArgumentParser(description="Seawen Qwen3.5-9B QLoRA")
    parser.add_argument("--model", default=os.environ.get("SEAWEN_MODEL", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--max-samples", type=int, default=int(os.environ.get("SEAWEN_MAX_SAMPLES", "4000")))
    parser.add_argument("--max-seq", type=int, default=int(os.environ.get("SEAWEN_MAX_SEQ", "2048")))
    parser.add_argument("--epochs", type=float, default=float(os.environ.get("SEAWEN_EPOCHS", "1")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("SEAWEN_LR", "1e-4")))
    parser.add_argument("--rank", type=int, default=int(os.environ.get("SEAWEN_RANK", "16")))
    parser.add_argument("--alpha", type=int, default=int(os.environ.get("SEAWEN_ALPHA", "32")))
    parser.add_argument("--batch", type=int, default=int(os.environ.get("SEAWEN_BATCH", "1")))
    parser.add_argument("--grad-accum", type=int, default=int(os.environ.get("SEAWEN_GRAD_ACCUM", "16")))
    parser.add_argument("--output", default=os.environ.get("SEAWEN_OUTPUT", "/kaggle/working/seawen-adapter"))
    parser.add_argument("--save-steps", type=int, default=int(os.environ.get("SEAWEN_SAVE_STEPS", "5")))
    parser.add_argument("--hub-id", default=os.environ.get("SEAWEN_HUB_ID", "Yashhh999/seawen"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dry-run", action="store_true", help="Build the mix and print two samples, then exit.")
    return parser.parse_known_args()[0]


def _headers(tokenizer) -> tuple[str, str]:
    probe = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "USER_PROBE"},
            {"role": "assistant", "content": "ASSISTANT_PROBE"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    user_at = probe.find("USER_PROBE")
    asst_at = probe.rfind("ASSISTANT_PROBE")
    user_head = probe[:user_at]
    asst_head = probe[user_at + len("USER_PROBE") : asst_at]
    # Keep the role marker, not the whole user turn.
    for token in ("<|im_start|>user", "<|start|>user", "<start_of_turn>user"):
        if token in user_head:
            user_head = user_head[user_head.rfind(token) :]
            break
    return user_head, asst_head


def main():
    args = parse_args()
    import unsloth  # before transformers / peft / huggingface_hub
    from seawen.hub import activate_token, get_token

    token = get_token()
    if not token:
        raise SystemExit(
            "No Hugging Face token. In Kaggle: Add-ons -> Secrets, "
            "label it HF_TOKEN, attach it to this notebook, then run again."
        )
    activate_token(token)

    from seawen.data import build_mix, render_text

    print(
        f"mix={args.max_samples} seq={args.max_seq} epochs={args.epochs} "
        f"lr={args.lr} rank={args.rank} save_steps={args.save_steps} "
        f"gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    if args.dry_run:
        rows = build_mix(args.max_samples, seed=args.seed)
        print(f"built {len(rows)} chats")
        for row in rows[:2]:
            print("---")
            print(row["messages"][1]["content"][:500])
        return

    import json
    import torch
    from unsloth import FastLanguageModel
    from transformers import TrainerCallback
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from seawen.hub import (
        pull_mix,
        push_adapter,
        push_checkpoint,
        push_mix,
        resolve_resume,
    )

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU. In Kaggle: Settings -> Accelerator -> GPU T4 x2, Internet On.")

    print(f"hub={args.hub_id} checkpoint every {args.save_steps} steps")

    out = args.output
    os.makedirs(out, exist_ok=True)
    mix_path = os.path.join(out, "mix.jsonl")
    if token and not os.path.isfile(mix_path):
        try:
            pull_mix(token, out, args.hub_id)
        except Exception as exc:
            print(f"mix download failed ({exc})")

    saved_texts = None
    if os.path.isfile(mix_path) and os.path.getsize(mix_path) > 0:
        saved_texts = []
        with open(mix_path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    saved_texts.append(json.loads(line)["text"])
        print(f"loaded {len(saved_texts)} saved rows from mix.jsonl")
    else:
        print("no saved mix, building a new one")

    resume = None
    try:
        resume = resolve_resume(out, token, args.hub_id)
    except Exception as exc:
        print(f"resume lookup failed, starting fresh ({exc})")
        resume = None

    major, minor = torch.cuda.get_device_capability()
    bf16 = major >= 8
    print(f"device={torch.cuda.get_device_name(0)} cc={major}.{minor} bf16={bf16}")
    print("Loading the base. On a T4 the Mamba kernels compile once and look stuck.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq,
        dtype=torch.bfloat16 if bf16 else torch.float16,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    if saved_texts is None:
        rows = build_mix(args.max_samples, seed=args.seed)
        texts = [render_text(tokenizer, row["messages"]) for row in rows]
        texts = [text for text in texts if text and text.strip()]
        with open(mix_path, "w") as handle:
            for text in texts:
                handle.write(json.dumps({"text": text}) + "\n")
        print(f"wrote {len(texts)} rows to {mix_path}")
        if token:
            try:
                push_mix(token, mix_path, args.hub_id)
            except Exception as exc:
                print(f"mix upload failed, training continues ({exc})")
    else:
        texts = saved_texts
    if resume and not texts:
        raise SystemExit(
            "A checkpoint exists but mix.jsonl is missing locally and on the hub. "
            "Refusing to resume onto a newly shuffled dataset. "
            "Delete the checkpoints on Yashhh999/seawen if you want a fresh run."
        )
    dataset = Dataset.from_dict({"text": texts})
    print("sample rendered tail:")
    print(texts[0][-400:])

    out = args.output
    config_kwargs = dict(
        output_dir=out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not bf16,
        bf16=bf16,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=args.seed,
        max_grad_norm=0.3,
        report_to="none",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        dataset_text_field="text",
        max_seq_length=args.max_seq,
        packing=False,
        dataset_num_proc=2,
    )
    try:
        sft_config = SFTConfig(**config_kwargs)
    except TypeError:
        config_kwargs.pop("dataset_text_field", None)
        config_kwargs.pop("max_seq_length", None)
        config_kwargs.pop("packing", None)
        config_kwargs.pop("dataset_num_proc", None)
        sft_config = SFTConfig(**config_kwargs)

    trainer_kwargs = dict(
        model=model,
        train_dataset=dataset,
        args=sft_config,
    )
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, dataset_text_field="text", max_seq_length=args.max_seq, **trainer_kwargs)

    try:
        from unsloth.chat_templates import train_on_responses_only

        user_part, asst_part = _headers(tokenizer)
        print(f"response-only mask user={user_part!r} assistant={asst_part!r}")
        trainer = train_on_responses_only(
            trainer,
            instruction_part=user_part,
            response_part=asst_part,
        )
    except Exception as exc:
        print(f"response-only mask skipped ({exc}). Loss will include the prompt.")

    class HubCheckpointCallback(TrainerCallback):
        def __init__(self, hf_token: str, repo_id: str):
            self.hf_token = hf_token
            self.repo_id = repo_id

        def on_save(self, args, state, control, **kwargs):
            folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if not os.path.isfile(os.path.join(folder, "trainer_state.json")):
                print(f"checkpoint-{state.global_step} incomplete, not pushing")
                return
            try:
                push_checkpoint(self.hf_token, folder, state.global_step, self.repo_id)
            except Exception as exc:
                print(f"checkpoint-{state.global_step} push failed, training continues ({exc})")

    if token:
        trainer.add_callback(HubCheckpointCallback(token, args.hub_id))

    try:
        trainer.train(resume_from_checkpoint=resume)
    except Exception as exc:
        text = str(exc).lower()
        if resume and ("checkpoint" in text or "resume" in text):
            print(f"resume from {resume} failed ({exc}). Starting at step 0 with the saved mix.")
            trainer.train(resume_from_checkpoint=None)
        else:
            raise
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    if token:
        try:
            push_adapter(token, out, args.hub_id)
        except Exception as exc:
            print(f"final adapter push failed ({exc})")
    print(f"adapter saved to {out}")


if __name__ == "__main__":
    main()

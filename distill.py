#!/usr/bin/env python3
"""Distill a large teacher into a small student.

MergeKit only averages tensors of identical shape, so it cannot move reasoning
across architectures. This does it through data instead: the teacher answers
prompts, then the student is LoRA-tuned to reason the same way, in the Thought
and Final Answer format the agent reads.

    python distill.py gen --prompts prompts.txt --limit 300
    python distill.py train
    python distill.py merge

Then follow "Export" in the README to turn out/pechi-merged into a GGUF.
"""

import argparse
import json
import shutil
from pathlib import Path

from pechi_agent import ollama, system_prompt

HERE = Path(__file__).parent
STUDENT = "huihui-ai/Huihui-Qwen3.5-9B-abliterated"
TEACHER = "qwen3.6-35b:latest"
IGNORED_LABEL = -100
LORA_TARGETS = r".*\.(self_attn\.[qkvo]_proj|linear_attn\.(in_proj_qkv|out_proj)|mlp\.(up|gate|down)_proj)"
# embed_tokens and lm_head are untied, unquantized and 2 GB each. With them on the
# GPU the 4-bit student needs about 7.5 GB, more than an 8 GB card has free.
DEVICE_MAP = {"model.embed_tokens": "cpu", "lm_head": "cpu", "": 0}


def read_rows(path: Path) -> list[dict]:
    """Rows from a JSONL file. A kill mid-write damages one row, not the whole run."""
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as lines:  # not splitlines: U+2028 is valid JSON
        for line in filter(str.strip, lines):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"skipping a damaged row in {path}")
    return rows


def gen(args: argparse.Namespace) -> None:
    """Teacher answers each prompt. Every row is flushed, so a rerun resumes."""
    lines = args.prompts.read_text(encoding="utf-8").splitlines()
    prompts = list(dict.fromkeys(line.strip() for line in lines if line.strip()))
    done = {row["prompt"] for row in read_rows(args.data)}
    todo = [p for p in prompts if p not in done][: args.limit]
    args.data.parent.mkdir(parents=True, exist_ok=True)
    options = {
        "num_gpu": args.num_gpu,
        "num_ctx": 4096,
        "num_predict": args.max_tokens,
        "temperature": 0.3,
    }
    with args.data.open("a", encoding="utf-8") as out:
        out.write("\n")  # never glue a new row onto a line a killed run left unfinished
        for number, prompt in enumerate(todo, 1):
            reply = ollama(
                "/api/chat",
                {
                    "model": args.teacher,
                    "stream": False,
                    "think": True,
                    "keep_alive": "10m",
                    "options": options,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            message = reply["message"]
            row = {
                "prompt": prompt,
                "thinking": message.get("thinking") or "",
                "content": message.get("content") or "",
                "done_reason": reply.get("done_reason", ""),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"{number}/{len(todo)}  {reply.get('eval_count')} tokens  {row['done_reason']}"
            )


def encode_rows(rows: list[dict], tokenizer, max_len: int) -> list[dict]:
    """Tokenise finished answers as the agent prompts them. Prompt tokens carry no loss."""
    system = system_prompt(Path("workspace"))
    usable = [
        r
        for r in rows
        if r["done_reason"] == "stop" and r["thinking"].strip() and r["content"].strip()
    ]
    encoded = []
    for row in usable:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": row["prompt"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        answer = f"Thought: {row['thinking'].strip()}\nFinal Answer: {row['content'].strip()}<|im_end|>\n"
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
        if (
            len(prompt_ids) + len(answer_ids) <= max_len
        ):  # a cut answer would teach it never to finish
            encoded.append(
                {
                    "input_ids": prompt_ids + answer_ids,
                    "labels": [IGNORED_LABEL] * len(prompt_ids) + answer_ids,
                }
            )
    print(f"training on {len(encoded)} of {len(rows)} rows")
    return encoded


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise SystemExit("Training needs a CUDA GPU, and PyTorch cannot see one.")
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    encoded = encode_rows(read_rows(args.data), tokenizer, args.max_len)
    if not encoded:
        raise SystemExit(
            f"No usable rows in {args.data}. Run gen first, or raise --max-len."
        )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.student,
        quantization_config=quantization,
        device_map=DEVICE_MAP,
        dtype=torch.bfloat16,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.05,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    settings = TrainingArguments(
        output_dir=str(args.adapter),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )
    # Batch size is 1, so the collator only has to add the batch dimension.
    collate = lambda batch: {
        key: torch.tensor([value]) for key, value in batch[0].items()
    }
    Trainer(
        model=model, args=settings, train_dataset=encoded, data_collator=collate
    ).train()
    model.save_pretrained(args.adapter)
    print(f"adapter saved to {args.adapter}")


def lora_pairs(adapter: Path) -> tuple[dict, float]:
    """Map checkpoint tensor names to (lora_B, lora_A), and return the LoRA scale."""
    from safetensors.torch import load_file

    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    lora = load_file(adapter / "adapter_model.safetensors")
    pairs = {}
    for key, down in lora.items():
        if ".lora_A." in key:
            module = key.removeprefix("base_model.model.").split(".lora_A.")[0]
            # Training used the text-only class; the checkpoint nests it one level down.
            name = module.replace("model.layers.", "model.language_model.layers.", 1)
            pairs[f"{name}.weight"] = (lora[key.replace(".lora_A.", ".lora_B.")], down)
    return pairs, config["lora_alpha"] / config["r"]


def merge(args: argparse.Namespace) -> None:
    """Bake the adapter into full weights, one shard at a time.

    Ollama no longer loads LoRA adapters, so the weights themselves must change.
    Working per shard keeps RAM near one 5 GB shard on a 19 GB model, and leaves
    every tensor in its stored dtype: this model keeps A_log and the linear_attn
    norms in F32, which a whole-model dtype= would round down.
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    student = Path(args.student)
    source = student if student.is_dir() else Path(snapshot_download(args.student))
    pairs, scale = lora_pairs(args.adapter)
    args.merged.mkdir(parents=True, exist_ok=True)
    for path in sorted(p for p in source.iterdir() if p.is_file()):
        if path.suffix != ".safetensors":
            shutil.copy2(path, args.merged)
            continue
        tensors = load_file(path)
        for name in tensors.keys() & pairs.keys():
            up, down = pairs.pop(name)
            baked = tensors[name].float() + scale * (up.float() @ down.float())
            tensors[name] = baked.to(tensors[name].dtype)
        save_file(tensors, args.merged / path.name, metadata={"format": "pt"})
        print(f"merged {path.name}")
    if pairs:
        raise SystemExit(
            f"{len(pairs)} adapter tensors matched nothing, such as {next(iter(pairs))}. "
            f"Do not use {args.merged}."
        )
    print(f"full model saved to {args.merged}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=HERE / "data" / "teacher.jsonl")
    parser.add_argument("--adapter", type=Path, default=HERE / "out" / "pechi-lora")
    parser.add_argument("--merged", type=Path, default=HERE / "out" / "pechi-merged")
    parser.add_argument("--student", default=STUDENT)
    commands = parser.add_subparsers(dest="command", required=True)

    teach = commands.add_parser("gen", help="teacher answers prompts")
    teach.add_argument(
        "--prompts", type=Path, required=True, help="text file, one prompt per line"
    )
    teach.add_argument("--limit", type=int, default=300)
    teach.add_argument("--teacher", default=TEACHER)
    teach.add_argument("--max-tokens", type=int, default=1024)
    teach.add_argument(
        "--num-gpu",
        type=int,
        default=26,
        help="layers on the GPU; 26 fits 8 GB, lower it if Ollama reports out of memory",
    )
    teach.set_defaults(run=gen)

    tune = commands.add_parser("train", help="LoRA-tune the student")
    tune.add_argument("--epochs", type=float, default=2)
    tune.add_argument("--lr", type=float, default=2e-4)
    tune.add_argument("--rank", type=int, default=16)
    tune.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help="lower this first if training runs out of VRAM",
    )
    tune.set_defaults(run=train)

    bake = commands.add_parser("merge", help="bake the LoRA into full weights")
    bake.set_defaults(run=merge)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()

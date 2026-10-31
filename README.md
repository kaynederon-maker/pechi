# Pechi

<img src="assets/pechi.svg" width="180" alt="Pechi, a round fawn Malinois puppy">

A little AI butler that lives in your terminal: fiercely loyal, sharp and quick,
like a well-trained Belgian Malinois in a very small, very round body. Pechi runs
on your own machine through Ollama. No cloud and no API keys. Its only connection
is to Ollama on this PC, plus the web tools when a task uses them.

| File | Purpose |
|---|---|
| `pechi_tui.py` | The `pechi` command: the interface, moods and animation. |
| `pechi.tcss` | Layout and motion. |
| `pechi_themes.py` | The colour themes. |
| `pechi_sprites.py` | Pechi as animated pixel art. |
| `pechi_agent.py` | The ReAct agent, its tools, and the switch between models. |
| `pechi_web.py` | Web search, page reading and scraping. |
| `install_pechi.py` | Installs the `pechi` command. |
| `assets/pechi.svg` | The mascot. |
| `distill.py` | Teacher-to-student distillation between the two models. |
| `Modelfile` | Imports a distilled model into Ollama. |
| `deploy_pechi.py` | This deployer. |
| `test_pechi.py` | Agent, distillation and deployment tests. |
| `test_pechi_tui.py` | Interface, web and model-switching tests. |
| `requirements.txt` | Python dependencies. |
| `.gitignore` | Keeps weights, data and logs out of git. |

## Quick start

Install [Ollama](https://ollama.com) and pull the two models. `MODELS` in
`pechi_agent.py` names them, and the 35B is a local import, so point that entry at
your own copy. Then:

```powershell
python -m venv --system-site-packages .venv
.venv\Scripts\activate
pip install -r requirements.txt
python install_pechi.py
```

Open a new PowerShell window, go to the folder you want to work in, and type:

```powershell
pechi
```

Pechi refuses to write into its own project folder, so run it from somewhere else.

## Using Pechi

| Key or command | What it does |
|---|---|
| `ctrl+t` or `/brain` | Switch between the 9B and 35B models, even in the middle of a task |
| `ctrl+g` or `/theme` | Change the theme, with a live preview |
| `/auto` | Turn continue all on or off |
| Click Pechi, `ctrl+p` or `/pet` | Pet Pechi. The ears, head, nose and belly each get their own reaction |
| `/boop` and `/treat` | Boop the nose, or hand over a treat |
| `ctrl+l` or `/clear` | Start a fresh chat |
| `pgup` and `pgdn` | Scroll the chat. New text follows again once you are back at the bottom |
| `esc` | Stop the current task. Say "continue" to pick it up |
| `ctrl+q` or `/quit` | Leave |

The themes are beige, milk, burgundy, cocoa, matcha and sakura. Your theme, model
and continue all choice are remembered in `~/.pechi/config.json`. Set
`TEXTUAL_ANIMATIONS=none` to turn the motion off.

Pechi can read and write files, run shell commands and Python, search the web,
read pages, and scrape them with a CSS selector. Pages that need JavaScript come
back thin.

There is no step limit. A task runs until it finishes, you press `esc`, or five
steps in a row fail, and every step's reasoning and tool work stays on screen. If
Ollama cuts a long reply off, Pechi resumes it by itself. A stopped task keeps its
work, so saying "continue" carries on from there.

## Why two models of different sizes cannot simply be merged

SLERP and TIES combine weights tensor by tensor, so both models need identical
shapes. Qwen3.5 9B (32 dense blocks, 4096 wide) and Qwen3.6 35B (40 blocks,
2048 wide, 256 experts) share shapes on only 4.54% of the small model's
parameters, about the size of its vision tower. MergeKit stops with
`Non-vocabulary dimensions must match`, even with `--allow-crimes`.

Moving reasoning from a large model into a small one is distillation, and
`distill.py` does that. Most people can skip it, because the toggle already gives
you both models.

### Distill

Takes hours and about 45 GB of free disk.

```powershell
python -c "from datasets import load_dataset as L; open('prompts.txt','w',encoding='utf8').write('\n'.join(q.replace('\n',' ') for q in L('openai/gsm8k','main',split='train[:300]')['question']))"
python distill.py gen --prompts prompts.txt --limit 3
python distill.py train --epochs 1
python distill.py gen --prompts prompts.txt
python distill.py train
python distill.py merge
```

The three-row pass proves the chain fits your hardware before you spend hours on
the teacher. `gen` resumes where it stopped, and rows already answered are kept.
If `train` runs out of VRAM, lower `--max-len`. It saves a checkpoint every 50
steps, so after a late crash either rerun it or use the latest checkpoint with
`python distill.py --adapter out\pechi-lora\checkpoint-50 merge`. There is no
resume: Trainer cannot reload an adapter onto the CPU-offloaded layers.
Install `flash-linear-attention` and `causal-conv1d` if they build on your system,
because transformers otherwise uses a much slower reference implementation.

### Export

```powershell
winget install ggml.llamacpp
git clone --depth 1 https://github.com/ggml-org/llama.cpp
python llama.cpp\convert_hf_to_gguf.py out\pechi-merged --outfile out\pechi-f16.gguf --outtype f16
llama-quantize out\pechi-f16.gguf out\pechi.Q4_K_M.gguf Q4_K_M
ollama create pechi-custom -f Modelfile
```

Then add `pechi-custom` to `MODELS` in `pechi_agent.py`, and the toggle will reach
it. Ollama's own `--quantize` has no GGUF `Q4_K_M` option, and it no longer loads
LoRA adapters. If disk is tight, the student's Hugging Face cache (19 GB) can go
once `distill.py merge` has finished. This GGUF is text only.

## Safety

- Listing and reading files in the current folder never asks, so keep secrets such
  as `.env` out of it. Writes, shell commands and Python each ask first.
- Opening a website asks first, and you can trust a site for the session. Private
  and local addresses are always refused, including through redirects.
- Continue all skips those prompts, except for calls matching a list of high-risk
  commands, which still need a human. Press `c` on an approval card or type `/auto`
  to keep it on across sessions, or pass `--auto` for one session. The top bar shows
  when it is on.
- File tools stay inside the folder, and `write_file` refuses Pechi's own code.
  `run_shell` and `run_python` have neither limit: they act with your full user
  rights. The list is a seatbelt, not a sandbox.
- Tools run without environment variables named like credentials, but they can
  still reach anything your user account can, such as `gh auth token`.
- The default models have their refusal layer removed, so the approval prompt is
  the brake. Keep it on, and use continue all inside a VM or container.
- Files and web pages can carry prompt injection, and the model may obey it.
- Each step is recorded in `pechi_audit.jsonl`, with output cut to 500 characters
  and known secret formats redacted.

## Credits

The agent loop follows the patterns of
[Kocoro-lab/Shannon](https://github.com/Kocoro-lab/Shannon) (MIT). The interface is
built with [Textual](https://github.com/Textualize/textual). Its mood is inspired
by soft, round slice-of-life characters such as Chiikawa, but Pechi is an original
character based on a real dog.

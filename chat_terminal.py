# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "jinja2>=3.1",
# ]
# ///
"""
Terminal chat with Gemma 4 E2B via litert_lm_advanced_main.exe.

Jinja2 formats the full conversation prompt using chat_template.jinja
before each turn; the binary handles inference.

Usage:
  uv run chat_terminal.py
  uv run chat_terminal.py --model gemma-4-E2B-it.litertlm --backend cpu
  uv run chat_terminal.py --max-tokens 8192
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_SCRIPT_DIR   = Path(__file__).parent
_TEMPLATE_FILE = _SCRIPT_DIR / "chat_template.jinja"
_DEFAULT_EXE   = _SCRIPT_DIR / "litert_lm_advanced_main.exe"
_DEFAULT_MODEL = _SCRIPT_DIR / "gemma-4-E2B-it.litertlm"

# ── Jinja2 chat template ──────────────────────────────────────────────────────
if not _TEMPLATE_FILE.exists():
    sys.exit(f"Missing {_TEMPLATE_FILE}")

_env = Environment(
    loader=FileSystemLoader(str(_SCRIPT_DIR)),
    keep_trailing_newline=True,
)
_tmpl = _env.get_template(_TEMPLATE_FILE.name)


def build_prompt(history: list[dict]) -> str:
    return _tmpl.render(
        messages=history,
        bos_token="",
        add_generation_prompt=True,
    )


# ── Parse stdout from litert_lm_main ─────────────────────────────────────────
def parse_response(stdout: str) -> tuple[str, dict]:
    """Return (reply_text, stats_dict) from the binary's stdout."""
    lines      = stdout.splitlines()
    reply_lines = []
    stats_lines = []
    in_stats   = False

    for line in lines:
        if line.startswith("input_prompt:"):
            continue
        if line.startswith("BenchmarkInfo:"):
            in_stats = True
        if in_stats:
            stats_lines.append(line)
        else:
            reply_lines.append(line)

    reply = "\n".join(reply_lines).strip()

    # Pull a couple of key numbers out of the stats block for display.
    stats: dict = {}
    for line in stats_lines:
        line = line.strip()
        if "Time to first token:" in line:
            try:
                stats["ttft"] = float(line.split(":")[1].replace("s", "").strip())
            except ValueError:
                pass
        if "Decode Speed:" in line:
            try:
                stats["tok_s"] = float(line.split(":")[1].replace("tokens/sec.", "").strip())
            except ValueError:
                pass
        if "Decode Turn" in line and "Processed" in line:
            try:
                stats["tokens"] = int(line.split("Processed")[1].split("tokens")[0].strip())
            except (ValueError, IndexError):
                pass

    return reply, stats


# ── Streaming inference ───────────────────────────────────────────────────────
_STOP = b"BenchmarkInfo:"
_LOOKAHEAD = len(_STOP) - 1  # bytes to hold back before printing


def infer(exe: Path, model: Path, backend: str, prompt: str,
          max_tokens: int = 4096) -> tuple[str, dict]:
    """Stream tokens to stdout as they arrive; return (full_reply, stats)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False, encoding="utf-8") as f:
        f.write(prompt)
        prompt_file = Path(f.name)

    try:
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            [str(exe),
             f"--model_path={model}",
             f"--input_prompt_file={prompt_file}",
             f"--backend={backend}",
             f"--max_num_tokens={max_tokens}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,          # no extra Python-side buffering
        )

        header_done = False     # consumed "input_prompt: ...\n"
        window      = b""      # lookahead buffer
        reply_bytes = b""      # full reply for history
        stats_bytes = b""      # everything from BenchmarkInfo: onwards

        while True:
            ch = proc.stdout.read(1)
            if not ch:
                # Process ended — flush remaining window as reply
                sys.stdout.buffer.write(window)
                sys.stdout.buffer.flush()
                reply_bytes += window
                break

            window += ch

            # Skip the echoed "input_prompt: ...\n" header line
            if not header_done:
                if b"\n" in window:
                    window = window[window.index(b"\n") + 1:]
                    header_done = True
                continue

            # Check if the stop marker has arrived in the window
            if _STOP in window:
                idx = window.index(_STOP)
                safe = window[:idx]
                sys.stdout.buffer.write(safe)
                sys.stdout.buffer.flush()
                reply_bytes += safe
                stats_bytes = window[idx:]
                break

            # Safe to emit everything except the last _LOOKAHEAD bytes
            if len(window) > _LOOKAHEAD:
                safe = window[:-_LOOKAHEAD]
                sys.stdout.buffer.write(safe)
                sys.stdout.buffer.flush()
                reply_bytes += safe
                window = window[-_LOOKAHEAD:]

        # Drain remaining stdout (rest of the stats block)
        stats_bytes += proc.stdout.read()
        proc.wait()
        elapsed = time.perf_counter() - t0

    finally:
        prompt_file.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise RuntimeError(stats_bytes.decode("utf-8", errors="replace")[:500])

    reply = reply_bytes.decode("utf-8", errors="replace").strip()
    _, stats = parse_response("BenchmarkInfo:\n" +
                              stats_bytes.decode("utf-8", errors="replace"))
    stats.setdefault("elapsed", round(elapsed, 1))
    return reply, stats


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",      default=str(_DEFAULT_MODEL), help="Path to .litertlm model")
    p.add_argument("--exe",        default=str(_DEFAULT_EXE),   help="Path to litert_lm_advanced_main.exe")
    p.add_argument("--backend",    default="cpu", choices=["cpu", "gpu"], help="Inference backend")
    p.add_argument("--max-tokens", default=4096, type=int,       help="KV-cache size (input+output)")
    args = p.parse_args()

    exe        = Path(args.exe)
    model      = Path(args.model)
    max_tokens = args.max_tokens

    for f in (exe, model, _TEMPLATE_FILE):
        if not f.exists():
            sys.exit(f"Not found: {f}")

    print(f"Model    : {model.name}")
    print(f"Backend  : {args.backend}")
    print(f"KV cache : {max_tokens} tokens")
    print("Type 'quit' or Ctrl-C to exit.\n")

    history: list[dict] = []

    while True:
        try:
            user_text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_text:
            continue
        if user_text.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        history.append({"role": "user", "content": user_text})
        prompt = build_prompt(history)

        print("Gemma: ", end="", flush=True)
        try:
            reply, stats = infer(exe, model, args.backend, prompt, max_tokens)
        except RuntimeError as e:
            print(f"\n[error] {e}")
            history.pop()
            continue

        print()  # newline after streamed tokens

        tok_s  = stats.get("tok_s")
        tokens = stats.get("tokens")
        ttft   = stats.get("ttft")
        parts  = []
        if tokens:  parts.append(f"~{tokens} tok")
        if tok_s:   parts.append(f"{tok_s:.1f} tok/s")
        if ttft:    parts.append(f"ttft {ttft:.2f}s")
        if parts:
            print(f"       \033[2m[{' · '.join(parts)}]\033[0m")
        print()

        history.append({"role": "model", "content": reply})


if __name__ == "__main__":
    main()

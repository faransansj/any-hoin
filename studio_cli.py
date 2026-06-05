#!/usr/bin/env python3
"""Local command-line tools for HoloScope Studio."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import termios
import time
import tty
import unicodedata
from pathlib import Path

from studio.doctor import count_by_status, current_device_info, doctor_json, run_doctor, system_info


PROJECT_ROOT = Path(__file__).resolve().parent
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_REVERSE = "\033[7m"
ANSI_CLEAR = "\033[2J\033[H"
COMMAND_SCHEMA_VERSION = 1
EXIT_CODES = {
    "0": "success",
    "1": "doctor found errors or command failed",
    "2": "safe describe-only mode; add --yes to execute",
    "127": "command executable not found",
    "130": "interrupted",
}


MenuEntry = tuple[str, str, str]


def _status_icon(status: str) -> str:
    return {"ok": "[OK]", "warn": "[WARN]", "error": "[ERROR]"}.get(status, "[?]")


def _print_doctor(result: dict) -> None:
    counts = count_by_status(result["checks"])
    device = result["device"]
    print(f"Doctor summary: {result['summary'].upper()}  ok={counts['ok']} warn={counts['warn']} error={counts['error']}")
    print(f"Selected device: {device['selected']} - {device['label']}")
    print(f"Torch: {device['torch_version']}  IPEX: {device['ipex_version'] or 'not installed'}")
    print()
    for check in result["checks"]:
        print(f"{_status_icon(check['status'])} {check['name']}: {check['detail']}")
        if check.get("suggestion"):
            print(f"    fix: {check['suggestion']}")


def cmd_doctor(args: argparse.Namespace) -> int:
    if args.json:
        print(doctor_json(PROJECT_ROOT))
        return 0
    result = run_doctor(PROJECT_ROOT)
    _print_doctor(result)
    return 1 if result["summary"] == "error" else 0


def cmd_device(args: argparse.Namespace) -> int:
    info = current_device_info()
    if args.json:
        import json

        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        print(f"Selected device: {info['selected']} - {info['label']}")
        print(f"Torch: {info['torch_version']}")
        print(f"IPEX: {info['ipex_version'] or 'not installed'}")
        print(f"XPU: build={info['xpu']['build']} available={info['xpu']['available']}")
        if info["xpu"].get("reason"):
            print(f"XPU reason: {info['xpu']['reason']}")
        print(f"CUDA available: {info['cuda_available']}")
        print(f"MPS available: {info['mps_available']}")
    return 0


def _log_path(action: str) -> Path:
    safe_action = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in action).strip("-")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = Path(tempfile.gettempdir()) / "any-hoin-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{safe_action or 'command'}-{stamp}.log"


def _execute_command(command: list[str], *, action: str = "command", stream_to_stderr: bool = False) -> dict:
    stream = sys.stderr if stream_to_stderr else sys.stdout
    path = _log_path(action)
    started = time.monotonic()
    print(f"$ {' '.join(command)}", file=stream, flush=True)
    print(f"[log] {path}", file=stream, flush=True)

    try:
        with path.open("w", encoding="utf-8", errors="replace") as log:
            log.write(f"$ {' '.join(command)}\n")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", file=stream, flush=True)
                log.write(line)
            returncode = process.wait()
    except KeyboardInterrupt:
        print("\n[interrupted]", file=stream, flush=True)
        returncode = 130
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=stream, flush=True)
        returncode = 127

    elapsed = round(time.monotonic() - started, 2)
    print(f"[done] exit={returncode} elapsed={elapsed}s log={path}", file=stream, flush=True)
    return {
        "action": action,
        "command": command,
        "cwd": str(PROJECT_ROOT),
        "returncode": returncode,
        "elapsed_sec": elapsed,
        "log_path": str(path),
    }


def _run_command(command: list[str], *, action: str = "command") -> int:
    return int(_execute_command(command, action=action)["returncode"])


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _command_payload(action: str, command: list[str], *, execute: bool) -> dict:
    return {
        "schema_version": COMMAND_SCHEMA_VERSION,
        "action": action,
        "cwd": str(PROJECT_ROOT),
        "command": command,
        "execute": execute,
    }


def _run_or_describe(action: str, command: list[str], args: argparse.Namespace) -> int:
    execute = bool(getattr(args, "yes", False))
    dry_run = bool(getattr(args, "dry_run", False))
    payload = _command_payload(action, command, execute=execute and not dry_run)

    json_mode = bool(getattr(args, "json", False))
    if json_mode and not execute:
        _print_json(payload)
    elif not json_mode:
        print("Command:", " ".join(command))

    if dry_run:
        return 0
    if not execute:
        if not json_mode:
            print("Add --yes to execute, or --dry-run for command generation only.")
        return 2
    result = _execute_command(command, action=action, stream_to_stderr=json_mode)
    if json_mode:
        _print_json({**payload, **result, "execute": True})
    return int(result["returncode"])


def build_crawl_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "crawling/danbooru_crawler.py",
        "--min-images",
        str(args.min_images),
        "--max-images",
        str(args.max_images),
        "--workers",
        str(args.workers),
        "--output-dir",
        args.output_dir,
    ]
    if getattr(args, "members", ""):
        command += ["--members", args.members]
    if getattr(args, "resize_large_images", False):
        command.append("--resize-large-images")
    return command


def build_train_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "train.py",
        "--data-dir",
        args.data_dir,
        "--save-dir",
        args.save_dir,
        "--batch-size",
        str(args.batch_size),
        "--phase1-epochs",
        str(args.phase1_epochs),
        "--phase2-epochs",
        str(args.phase2_epochs),
    ]
    if args.device and args.device != "auto":
        if args.device == "cpu":
            command.append("--cpu")
        elif args.device == "xpu":
            command.append("--xpu")
        else:
            command += ["--device", args.device]
    if args.mode == "fresh":
        command.append("--fresh")
    elif args.mode == "finetune":
        command.append("--finetune")
    return command


def cmd_crawl(args: argparse.Namespace) -> int:
    return _run_or_describe("crawl", build_crawl_command(args), args)


def cmd_train(args: argparse.Namespace) -> int:
    return _run_or_describe("train", build_train_command(args), args)


def cmd_backend_profile(args: argparse.Namespace) -> int:
    command = [sys.executable, "scripts/sync_backend.py", args.backend, "--apply", "--check"]
    execute = bool(args.yes) and not bool(args.dry_run)
    payload = _command_payload("backend-profile", command, execute=execute)

    if args.json and not execute:
        _print_json(payload)
    elif not args.json:
        print("Command:", " ".join(command))

    if args.dry_run:
        return 0
    if not args.yes:
        if not args.json:
            print("Add --yes to execute, or --dry-run for command generation only.")
        return 2

    result = _execute_command(command, action="backend-profile", stream_to_stderr=args.json)
    doctor_result = None if args.skip_doctor else run_doctor(PROJECT_ROOT)
    final = {**payload, **result, "execute": True}
    if doctor_result is not None:
        final["doctor_summary"] = doctor_result["summary"]
        final["doctor_counts"] = count_by_status(doctor_result["checks"])

    if args.json:
        _print_json(final)
    elif doctor_result is not None:
        print()
        print("[post-check] Doctor after backend sync:")
        _print_doctor(doctor_result)

    if int(result["returncode"]) != 0:
        return int(result["returncode"])
    if doctor_result is not None and doctor_result["summary"] == "error":
        return 1
    return 0


def cmd_commands(args: argparse.Namespace) -> int:
    commands = [
        {
            "name": "doctor",
            "description": "Run health checks",
            "json": True,
            "example": [sys.executable, "studio_cli.py", "doctor", "--json"],
        },
        {
            "name": "device",
            "description": "Show selected compute device",
            "json": True,
            "example": [sys.executable, "studio_cli.py", "device", "--json"],
        },
        {
            "name": "crawl",
            "description": "Build or run crawler command",
            "json": True,
            "safe_default": "describe-only",
            "example": [sys.executable, "studio_cli.py", "crawl", "--dry-run", "--json", "--members", "gawr_gura"],
        },
        {
            "name": "train",
            "description": "Build or run training command",
            "json": True,
            "safe_default": "describe-only",
            "example": [sys.executable, "studio_cli.py", "train", "--dry-run", "--json", "--device", "xpu"],
        },
        {
            "name": "backend-profile",
            "description": "Sync CPU/CUDA/ROCm/Arc backend and optionally re-run doctor",
            "json": True,
            "safe_default": "describe-only",
            "example": [sys.executable, "studio_cli.py", "backend-profile", "arc", "--dry-run", "--json"],
        },
        {"name": "tui", "description": "Open interactive menu", "json": False},
    ]
    if args.json:
        _print_json({"schema_version": COMMAND_SCHEMA_VERSION, "exit_codes": EXIT_CODES, "commands": commands})
    else:
        for item in commands:
            print(f"{item['name']}: {item['description']}")
    return 0


def _pause() -> None:
    if not sys.stdin.isatty():
        return
    try:
        input("\nPress Enter to return to the menu...")
    except EOFError:
        pass


def _save_doctor_report() -> Path:
    path = Path(tempfile.gettempdir()) / "any-hoin-doctor.json"
    path.write_text(doctor_json(PROJECT_ROOT) + "\n", encoding="utf-8")
    return path


def _read_key() -> str:
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        ch += sys.stdin.read(2)
    return ch


def _terminal_width() -> int:
    columns = shutil.get_terminal_size((88, 24)).columns
    return max(52, min(100, columns))


def _border(width: int, left: str = "+", fill: str = "-", right: str = "+") -> str:
    return left + (fill * (width - 2)) + right


def _cell_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def _fit_cells(text: str, width: int) -> str:
    result = []
    used = 0
    for char in text:
        if unicodedata.combining(char):
            char_width = 0
        else:
            char_width = 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
        if used + char_width > width:
            break
        result.append(char)
        used += char_width
    return "".join(result) + (" " * max(0, width - used))


def _boxed_line(text: str, width: int, style: str = "") -> str:
    inner = _fit_cells(text, width - 4)
    if style and sys.stdout.isatty() and "NO_COLOR" not in os.environ:
        inner = f"{style}{inner}{ANSI_RESET}"
    return "| " + inner + " |"


def _device_summary() -> list[str]:
    try:
        info = current_device_info()
        system = system_info()
    except Exception as exc:
        return [f"Device: unknown ({exc})"]
    return [
        f"OS: {system['system']} {system['release']} | CPU: {system['cpu_count']} cores | RAM: {system['ram_total']}",
        f"Device: {info['selected']} - {info['label']}",
        f"Torch: {info['torch_version']} | IPEX: {info['ipex_version'] or 'not installed'}",
    ]


def _render_menu(
    title: str,
    subtitle: str,
    entries: list[MenuEntry],
    selected: int,
    status_lines: list[str],
) -> None:
    width = _terminal_width()
    print(ANSI_CLEAR if sys.stdout.isatty() else "", end="")
    print(_border(width))
    print(_boxed_line("ANY-HOIN STUDIO", width, ANSI_BOLD))
    print(_boxed_line("HoloScope local control console", width, ANSI_DIM))
    print(_border(width))
    print(_boxed_line(title, width, ANSI_BOLD))
    if subtitle:
        print(_boxed_line(subtitle, width, ANSI_DIM))
    print(_border(width))
    for line in status_lines:
        print(_boxed_line(line, width))
    print(_border(width))
    print(_boxed_line("Actions", width))
    print(_border(width, "+", "-", "+"))

    label_width = max(_cell_width(label) for _, label, _ in entries)
    for idx, (key, label, description) in enumerate(entries):
        marker = ">" if idx == selected else " "
        row = f"{marker} [{key}] {_fit_cells(label, label_width)}  {description}"
        print(_boxed_line(row, width, ANSI_REVERSE if idx == selected else ""))

    print(_border(width))
    print(_boxed_line("Up/Down or j/k: move | Enter: select | c or /: command | q: quit/back", width))
    print(_border(width))


def _prompt_command() -> str:
    if sys.stdin.isatty():
        print()
    try:
        return input("command> ").strip()
    except EOFError:
        return ""


def _prompt_value(label: str, default: str) -> str:
    try:
        value = input(f"{label} [{default}]: ").strip()
    except EOFError:
        return default
    return value or default


def _confirm(message: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    try:
        value = input(f"{message} [{suffix}]: ").strip().lower()
    except EOFError:
        return default
    if not value:
        return default
    return value in {"y", "yes"}


def _run_crawl() -> int | None:
    print()
    print("Crawl images")
    min_images = _prompt_value("Minimum images per character", "500")
    max_images = _prompt_value("Maximum images per character", "1000")
    workers = _prompt_value("Download workers", "4")
    members = _prompt_value("Character keys, comma-separated (blank = all)", "")
    output_dir = _prompt_value("Output directory", "./dataset/raw")

    command = build_crawl_command(
        argparse.Namespace(
            min_images=min_images,
            max_images=max_images,
            workers=workers,
            members=members,
            output_dir=output_dir,
            resize_large_images=_confirm("Resize large images before saving", True),
        )
    )

    print()
    print("Command:", " ".join(command))
    if not _confirm("Start crawl", False):
        print("[OK] Cancelled.")
        return None
    return _run_command(command, action="crawl")


def _run_train() -> int | None:
    print()
    print("Train model")
    data_dir = _prompt_value("Dataset directory", "./dataset/raw")
    save_dir = _prompt_value("Checkpoint directory", "./checkpoints")
    device = _prompt_value("Device (auto, xpu, cuda, mps, cpu)", "auto")
    batch_size = _prompt_value("Batch size", "32")
    phase1 = _prompt_value("Phase 1 epochs", "5")
    phase2 = _prompt_value("Phase 2 epochs", "30")
    mode = _prompt_value("Mode (resume, finetune, fresh)", "resume")

    command = build_train_command(
        argparse.Namespace(
            data_dir=data_dir,
            save_dir=save_dir,
            device=device,
            batch_size=batch_size,
            phase1_epochs=phase1,
            phase2_epochs=phase2,
            mode=mode,
        )
    )

    print()
    print("Command:", " ".join(command))
    if not _confirm("Start training", False):
        print("[OK] Cancelled.")
        return None
    return _run_command(command, action="train")


def _print_help() -> None:
    print()
    print("TUI help")
    print()
    print("Navigation")
    print("  Up/Down or j/k  Move selection")
    print("  Enter           Run selected action")
    print("  1-7             Run numbered action")
    print("  c or /          Open command prompt")
    print("  q               Quit or go back")
    print()
    print("Commands")
    commands = [
        ("help, ?", "Show this command list"),
        ("crawl", "Start image crawling prompts"),
        ("train", "Start training prompts"),
        ("doctor", "Run health checks"),
        ("device", "Show selected compute device"),
        ("backend", "Open backend profile setup"),
        ("report", "Save doctor JSON report"),
        ("web", "Start Web Studio"),
        ("backend-server", "Run FastAPI backend only"),
        ("frontend-server", "Run Vite frontend only"),
        ("q, quit, exit", "Quit the TUI"),
        ("<shell command>", "Run a one-off command"),
    ]
    width = max(len(command) for command, _ in commands)
    for command, description in commands:
        print(f"  {command.ljust(width)}  {description}")


def _handle_command(raw: str) -> int | None:
    command = raw.strip()
    if not command:
        return None
    if command in {"help", "?"}:
        _print_help()
        _pause()
        return None
    if command == "crawl":
        result = _run_crawl()
        _pause()
        return result
    if command == "train":
        result = _run_train()
        _pause()
        return result
    if command == "doctor":
        _print_doctor(run_doctor(PROJECT_ROOT))
        _pause()
        return None
    if command == "device":
        cmd_device(argparse.Namespace(json=False))
        _pause()
        return None
    if command == "backend":
        _run_backend_setup()
        _pause()
        return None
    if command == "report":
        path = _save_doctor_report()
        print(f"[OK] Wrote {path}")
        _pause()
        return None
    if command == "web":
        return _run_command([sys.executable, "start.py", "--web"], action="web")
    if command == "backend-server":
        return _run_command([sys.executable, "studio_api.py"], action="backend-server")
    if command == "frontend-server":
        return _run_command(["npm", "run", "dev", "--prefix", "frontend"], action="frontend-server")
    if command in {"q", "quit", "exit"}:
        return 0
    _run_command(shlex.split(command), action="manual-command")
    _pause()
    return None


def _select_menu(title: str, subtitle: str, entries: list[MenuEntry]) -> str:
    if not sys.stdin.isatty():
        print()
        print("== ANY-HOIN STUDIO ==")
        print(f"-- {title} --")
        if subtitle:
            print(subtitle)
        label_width = max(_cell_width(label) for _, label, _ in entries)
        for key, label, description in entries:
            print(f"{key}. {_fit_cells(label, label_width)}  {description}")
        print("c or /. Command prompt")
        try:
            return input("> ").strip().lower()
        except EOFError:
            return "q"

    selected = 0
    status_lines = _device_summary()
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            _render_menu(title, subtitle, entries, selected, status_lines)
            key = _read_key()
            if key in {"\x1b[A", "k"}:
                selected = (selected - 1) % len(entries)
            elif key in {"\x1b[B", "j"}:
                selected = (selected + 1) % len(entries)
            elif key in {"\r", "\n"}:
                print()
                return entries[selected][0]
            elif key in {"c", "/"}:
                return "__command__"
            elif key.lower() in {entry[0] for entry in entries}:
                print()
                return key.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _backend_menu() -> str:
    return _select_menu(
        "Backend profile",
        "Choose the PyTorch accelerator profile to install and verify.",
        [
            ("1", "Auto detect", "Pick a likely backend for this machine"),
            ("2", "CPU", "Portable baseline, no GPU packages"),
            ("3", "MPS", "Apple Silicon backend"),
            ("4", "CUDA", "NVIDIA GPU backend"),
            ("5", "ROCm", "AMD GPU backend"),
            ("6", "Arc XPU", "Intel Arc/B-series XPU backend"),
            ("q", "Back", "Return to main menu"),
        ],
    )


def _run_backend_setup() -> int:
    mapping = {
        "1": "auto",
        "2": "cpu",
        "3": "mps",
        "4": "cuda",
        "5": "rocm",
        "6": "arc",
    }
    choice = _backend_menu()
    if choice in {"q", "quit", "exit"}:
        return 0
    backend = mapping.get(choice)
    if backend is None:
        print("[WARN] Unknown backend selection.")
        return 0
    return _run_command([sys.executable, "scripts/sync_backend.py", backend, "--apply", "--check"], action="backend-profile")


def _tui_menu() -> str:
    entries = [
        ("1", "Web Studio", "Start backend and Vite frontend"),
        ("2", "Crawl images", "Collect Danbooru images into dataset/raw"),
        ("3", "Train model", "Run train.py with guided options"),
        ("4", "Doctor", "Run health checks with fix hints"),
        ("5", "Accelerator status", "Show torch/XPU/CUDA/MPS status"),
        ("6", "Device/backend setup", "Install/check CPU, CUDA, ROCm, Arc"),
        ("7", "Save doctor report", "Write JSON report to /tmp"),
        ("q", "Quit", "Exit the TUI"),
    ]
    return _select_menu("Main menu", "Crawl -> Label -> Train -> Export -> Inference", entries)


def cmd_tui(args: argparse.Namespace) -> int:
    while True:
        choice = _tui_menu()
        if choice in {"q", "quit", "exit"}:
            return 0
        if choice in {"__command__", "c", "/"}:
            result = _handle_command(_prompt_command())
            if result is not None:
                return result
            continue
        if choice == "1":
            return _run_command([sys.executable, "start.py", "--web"], action="web")
        if choice == "2":
            result = _run_crawl()
            _pause()
            if result is not None:
                return result
            continue
        if choice == "3":
            result = _run_train()
            _pause()
            if result is not None:
                return result
            continue
        if choice == "4":
            _print_doctor(run_doctor(PROJECT_ROOT))
            _pause()
            continue
        if choice == "5":
            cmd_device(argparse.Namespace(json=False))
            _pause()
            continue
        if choice == "6":
            _run_backend_setup()
            _pause()
            continue
        if choice == "7":
            path = _save_doctor_report()
            print(f"[OK] Wrote {path}")
            _pause()
            continue
        print("[WARN] Unknown selection.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HoloScope Studio CLI")
    subparsers = parser.add_subparsers(dest="command")

    commands = subparsers.add_parser("commands", help="list machine-usable CLI commands")
    commands.add_argument("--json", action="store_true", help="print machine-readable JSON")
    commands.set_defaults(func=cmd_commands)

    doctor = subparsers.add_parser("doctor", help="run local health checks")
    doctor.add_argument("--json", action="store_true", help="print machine-readable JSON")
    doctor.set_defaults(func=cmd_doctor)

    device = subparsers.add_parser("device", help="show selected compute device")
    device.add_argument("--json", action="store_true", help="print machine-readable JSON")
    device.set_defaults(func=cmd_device)

    tui = subparsers.add_parser("tui", help="open the lightweight terminal menu")
    tui.set_defaults(func=cmd_tui)

    crawl = subparsers.add_parser("crawl", help="build or run the crawler command")
    crawl.add_argument("--min-images", type=int, default=500)
    crawl.add_argument("--max-images", type=int, default=1000)
    crawl.add_argument("--workers", type=int, default=4)
    crawl.add_argument("--members", default="", help="comma-separated character keys; blank means all")
    crawl.add_argument("--output-dir", default="./dataset/raw")
    crawl.add_argument("--resize-large-images", action="store_true")
    crawl.add_argument("--dry-run", action="store_true", help="only print the command")
    crawl.add_argument("--yes", action="store_true", help="execute the command")
    crawl.add_argument("--json", action="store_true", help="print machine-readable command payload")
    crawl.set_defaults(func=cmd_crawl)

    train = subparsers.add_parser("train", help="build or run the training command")
    train.add_argument("--data-dir", default="./dataset/raw")
    train.add_argument("--save-dir", default="./checkpoints")
    train.add_argument("--device", default="auto", choices=["auto", "cpu", "xpu", "cuda", "mps"])
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--phase1-epochs", type=int, default=5)
    train.add_argument("--phase2-epochs", type=int, default=30)
    train.add_argument("--mode", default="resume", choices=["resume", "finetune", "fresh"])
    train.add_argument("--dry-run", action="store_true", help="only print the command")
    train.add_argument("--yes", action="store_true", help="execute the command")
    train.add_argument("--json", action="store_true", help="print machine-readable command payload")
    train.set_defaults(func=cmd_train)

    backend = subparsers.add_parser("backend-profile", help="build or run backend profile sync")
    backend.add_argument("backend", choices=["auto", "cpu", "mps", "cuda", "rocm", "arc"])
    backend.add_argument("--dry-run", action="store_true", help="only print the command")
    backend.add_argument("--yes", action="store_true", help="execute the command")
    backend.add_argument("--skip-doctor", action="store_true", help="skip post-sync doctor verification")
    backend.add_argument("--json", action="store_true", help="print machine-readable command payload")
    backend.set_defaults(func=cmd_backend_profile)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

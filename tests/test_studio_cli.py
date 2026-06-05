import argparse
import json
import sys

import studio_cli


def test_build_crawl_command_for_agent():
    command = studio_cli.build_crawl_command(
        argparse.Namespace(
            min_images=100,
            max_images=200,
            workers=2,
            members="foo,bar",
            output_dir="./dataset/raw",
            resize_large_images=True,
        )
    )

    assert command[1:] == [
        "crawling/danbooru_crawler.py",
        "--min-images",
        "100",
        "--max-images",
        "200",
        "--workers",
        "2",
        "--output-dir",
        "./dataset/raw",
        "--members",
        "foo,bar",
        "--resize-large-images",
    ]


def test_build_train_command_for_xpu():
    command = studio_cli.build_train_command(
        argparse.Namespace(
            data_dir="./dataset/raw",
            save_dir="./checkpoints",
            device="xpu",
            batch_size=16,
            phase1_epochs=1,
            phase2_epochs=2,
            mode="fresh",
        )
    )

    assert "--xpu" in command
    assert "--fresh" in command
    assert command[1:4] == ["train.py", "--data-dir", "./dataset/raw"]


def test_commands_json_lists_agent_safe_commands(capsys):
    code = studio_cli.main(["commands", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["schema_version"] == 1
    assert payload["exit_codes"]["2"].startswith("safe")
    assert {item["name"] for item in payload["commands"]} >= {
        "doctor",
        "device",
        "crawl",
        "train",
        "backend-profile",
    }


def test_crawl_dry_run_json_does_not_execute(capsys):
    code = studio_cli.main(["crawl", "--dry-run", "--json", "--members", "foo"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["action"] == "crawl"
    assert payload["schema_version"] == 1
    assert payload["execute"] is False
    assert "--members" in payload["command"]


def test_backend_profile_runs_post_doctor(monkeypatch, capsys):
    monkeypatch.setattr(
        studio_cli,
        "_execute_command",
        lambda command, *, action, stream_to_stderr=False: {
            "action": action,
            "command": command,
            "cwd": str(studio_cli.PROJECT_ROOT),
            "returncode": 0,
            "elapsed_sec": 0.01,
            "log_path": "/tmp/test.log",
        },
    )
    monkeypatch.setattr(
        studio_cli,
        "run_doctor",
        lambda root: {
            "summary": "ok",
            "checks": [{"status": "ok"}],
            "device": {"selected": "xpu", "label": "test", "torch_version": "2.8.0+xpu", "ipex_version": "2.8.0+xpu"},
        },
    )

    code = studio_cli.main(["backend-profile", "arc", "--yes", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["action"] == "backend-profile"
    assert payload["doctor_summary"] == "ok"


def test_backend_profile_fails_on_doctor_error(monkeypatch, capsys):
    monkeypatch.setattr(
        studio_cli,
        "_execute_command",
        lambda command, *, action, stream_to_stderr=False: {
            "action": action,
            "command": command,
            "cwd": str(studio_cli.PROJECT_ROOT),
            "returncode": 0,
            "elapsed_sec": 0.01,
            "log_path": "/tmp/test.log",
        },
    )
    monkeypatch.setattr(
        studio_cli,
        "run_doctor",
        lambda root: {
            "summary": "error",
            "checks": [{"status": "error"}],
            "device": {"selected": "cpu", "label": "test", "torch_version": "bad", "ipex_version": "bad"},
        },
    )

    code = studio_cli.main(["backend-profile", "arc", "--yes", "--json"])

    assert code == 1


def test_execute_command_captures_log():
    result = studio_cli._execute_command(
        [sys.executable, "-c", "print('agent-log-ok')"],
        action="test-agent-runner",
    )

    assert result["returncode"] == 0
    assert result["elapsed_sec"] >= 0
    with open(result["log_path"], encoding="utf-8") as f:
        assert "agent-log-ok" in f.read()

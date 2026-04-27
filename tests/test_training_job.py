import asyncio
import json

from studio.jobs.base_job import _clean_output_line
from studio.jobs.train_job import TRAIN_EVENT_PREFIX, TrainJob


def test_train_job_parses_structured_progress_and_metrics():
    job = TrainJob()

    progress = {
        "split": "train",
        "pct": 25,
        "batch_cur": 1,
        "batch_total": 4,
        "eta_sec": 3.2,
        "speed_it_s": 1.25,
        "avg_loss": 0.9,
        "avg_acc": 0.25,
    }
    asyncio.run(
        job._on_line(
            "  train:  25%|##        | 1/4 [00:01<00:03,  1.00it/s]"
            + TRAIN_EVENT_PREFIX
            + json.dumps({"type": "progress", "data": progress}, ensure_ascii=False)
        )
    )

    assert job.current_progress == progress

    metric = {
        "phase": 1,
        "epoch": 1,
        "total_epochs": 2,
        "train_loss": 0.7,
        "train_acc": 0.5,
        "val_loss": 0.6,
        "val_acc": 0.75,
    }
    asyncio.run(
        job._on_line(
            TRAIN_EVENT_PREFIX
            + json.dumps({"type": "metric", "data": metric}, ensure_ascii=False)
        )
    )

    assert job.metrics == [metric]
    assert job.best_val_acc == 0.75

    # 같은 epoch 이벤트가 중복 도착해도 backend epoch_count가 늘어나면 안 된다.
    asyncio.run(
        job._on_line(
            TRAIN_EVENT_PREFIX
            + json.dumps({"type": "metric", "data": metric}, ensure_ascii=False)
        )
    )
    assert len(job.metrics) == 1


def test_train_job_hides_structured_and_tqdm_log_lines():
    job = TrainJob()

    assert job._format_log_line(TRAIN_EVENT_PREFIX + "{}") is None
    assert job._format_log_line("  train:  50%|█████     | 1/2 [00:01<00:01,  1.00it/s]" + TRAIN_EVENT_PREFIX + "{}") is None
    assert job._format_log_line("  train:  50%|█████     | 1/2 [00:01<00:01,  1.00it/s]") is None
    assert job._format_log_line("  warnings.warn(str(msg))") is None
    assert job._format_log_line("  warnings.warn(") is None
    assert job._format_log_line("  Epoch  1/2 | train_loss: 0.7000  train_acc: 0.5000 | val_loss: 0.6000  val_acc: 0.7500")


def test_clean_output_line_removes_carriage_return_progress_fragments():
    raw = "old progress\r  train: 100%|##########| 2/2 [00:01<00:00,  2.00it/s]\n"

    assert _clean_output_line(raw) == "  train: 100%|##########| 2/2 [00:01<00:00,  2.00it/s]"

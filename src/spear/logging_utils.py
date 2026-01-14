
import csv
import logging
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class _MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int) -> None:
        super().__init__()
        self._max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self._max_level


def configure_logging(logs_dir: Path, run_name: str, log_level: int = logging.INFO) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(_MaxLevelFilter(logging.INFO))
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.addHandler(file_handler)
    root.addHandler(stdout_handler)
    root.addHandler(stderr_handler)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)

    # Emit an initial run-context record so the file log contains node/time info
    try:
        cwd = os.getcwd()
        host = socket.gethostname()
        user = os.getenv("USER") or os.getenv("LOGNAME") or ""
        slurm_job = os.getenv("SLURM_JOB_ID")
        slurm_task = os.getenv("SLURM_ARRAY_TASK_ID")
        timestamp = datetime.utcnow().isoformat() + "Z"
        root.info(
            "Run context | run_name=%s | user=%s | cwd=%s | host=%s | timestamp_utc=%s | slurm_job=%s | slurm_task=%s",
            run_name,
            user,
            cwd,
            host,
            timestamp,
            slurm_job,
            slurm_task,
        )
    except Exception:
        # Never fail logging setup due to environment introspection
        pass

    return log_path


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name if name else __name__)


try:  # Optional dependency for resource tracking
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

try:  # Optional GPU visibility
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


@dataclass
class _ResourceSample:
    time_sec: float
    rss_gib: float
    cpu_percent: float
    thread_count: int
    gpu_memory_gib: float
    gpu_reserved_gib: float


class ResourceUsageTracker:
    """Best-effort background sampler that records resource usage and writes plots/csv."""

    def __init__(self, name: str, output_dir: Path | str, interval_seconds: float = 60.0) -> None:
        self._name = name
        self._safe_name = name.replace(" ", "_")
        self._output_dir = Path(output_dir).expanduser().resolve()
        self._interval = max(1.0, float(interval_seconds))
        self._log = logging.getLogger(__name__)
        self._enabled = psutil is not None
        self._records: List[_ResourceSample] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._process = psutil.Process() if psutil is not None else None

    def __enter__(self) -> "ResourceUsageTracker":
        if not self._enabled or self._process is None:
            self._log.debug("ResourceUsageTracker disabled (psutil unavailable)")
            return self
        self._start_time = time.time()
        try:
            self._process.cpu_percent(interval=None)
        except Exception:  # pragma: no cover - psutil quirks
            pass
        self._thread = threading.Thread(
            target=self._run,
            name=f"ResourceUsageTracker[{self._safe_name}]",
            daemon=True,
        )
        self._thread.start()
        self._log.debug("ResourceUsageTracker started for %s", self._name)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._enabled or self._process is None:
            return
        self._stop_event.set()
        try:
            self._sample_once()
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=self._interval)
        if not self._records:
            self._log.debug("No resource samples captured for %s", self._name)
            return
        self._write_csv()
        self._write_plot()

    def _run(self) -> None:
        if self._process is None:
            return
        while not self._stop_event.is_set():
            try:
                self._sample_once()
            except Exception:  # pragma: no cover - sampling should not crash
                self._log.debug("Resource sample failed for %s", self._name, exc_info=True)
            if self._stop_event.wait(self._interval):
                break

    def _sample_once(self) -> None:
        if self._process is None or self._start_time is None:
            return
        now = time.time()
        rel = now - self._start_time
        try:
            rss_bytes = self._process.memory_info().rss
        except Exception:
            rss_bytes = float("nan")
        try:
            cpu_percent = self._process.cpu_percent(interval=None)
        except Exception:
            cpu_percent = float("nan")
        try:
            thread_count = self._process.num_threads()
        except Exception:
            thread_count = -1

        gpu_memory = 0.0
        gpu_reserved = 0.0
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    for idx in range(torch.cuda.device_count()):
                        gpu_memory += float(torch.cuda.memory_allocated(idx)) / (1024 ** 3)
                        gpu_reserved += float(torch.cuda.memory_reserved(idx)) / (1024 ** 3)
            except Exception:  # pragma: no cover - defensive GPU handling
                pass

        self._records.append(
            _ResourceSample(
                time_sec=float(rel),
                rss_gib=float(rss_bytes) / (1024 ** 3) if rss_bytes == rss_bytes else float("nan"),
                cpu_percent=float(cpu_percent),
                thread_count=int(thread_count),
                gpu_memory_gib=float(gpu_memory),
                gpu_reserved_gib=float(gpu_reserved),
            )
        )

    def _write_csv(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / f"{self._safe_name}_resource_usage.csv"
        try:
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "time_sec",
                        "rss_gib",
                        "cpu_percent",
                        "thread_count",
                        "gpu_memory_gib",
                        "gpu_reserved_gib",
                    ]
                )
                for sample in self._records:
                    writer.writerow(
                        [
                            f"{sample.time_sec:.2f}",
                            f"{sample.rss_gib:.4f}",
                            f"{sample.cpu_percent:.2f}",
                            sample.thread_count,
                            f"{sample.gpu_memory_gib:.4f}",
                            f"{sample.gpu_reserved_gib:.4f}",
                        ]
                    )
            self._log.info("Wrote resource usage CSV to %s", path)
        except Exception:  # pragma: no cover - IO errors shouldn't crash pipeline
            self._log.warning("Failed to write resource usage CSV to %s", path, exc_info=True)

    def _write_plot(self) -> None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:  # pragma: no cover - matplotlib optional
            self._log.debug("Skipping resource usage plot for %s (matplotlib unavailable)", self._name)
            return

        times = [sample.time_sec for sample in self._records]
        rss = [sample.rss_gib for sample in self._records]
        cpu = [sample.cpu_percent for sample in self._records]
        gpu = [sample.gpu_memory_gib for sample in self._records]

        if not times:
            return

        fig, ax_left = plt.subplots(figsize=(8, 4.5))
        ax_left.plot(times, rss, label="RSS (GiB)", color="#1f77b4")
        ax_left.set_xlabel("Time (s)")
        ax_left.set_ylabel("Memory (GiB)", color="#1f77b4")
        ax_left.tick_params(axis="y", labelcolor="#1f77b4")

        ax_right = ax_left.twinx()
        ax_right.plot(times, cpu, label="CPU %", color="#ff7f0e", linestyle="--")
        ax_right.set_ylabel("CPU usage (%)", color="#ff7f0e")
        ax_right.tick_params(axis="y", labelcolor="#ff7f0e")

        if any(value > 0.0 for value in gpu):
            ax_right.plot(times, gpu, label="GPU mem (GiB)", color="#2ca02c", linestyle=":")

        ax_left.set_title(f"Resource usage | {self._name}")
        fig.tight_layout()

        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / f"{self._safe_name}_resource_usage.png"
        try:
            fig.savefig(path, dpi=150)
            self._log.info("Wrote resource usage plot to %s", path)
        except Exception:  # pragma: no cover - IO errors shouldn't crash pipeline
            self._log.warning("Failed to write resource usage plot to %s", path, exc_info=True)
        finally:
            plt.close(fig)

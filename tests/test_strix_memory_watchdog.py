#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import fcntl
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "strix_memory_watchdog.py"
)
SPEC = importlib.util.spec_from_file_location(
    "strix_memory_watchdog", SCRIPT_PATH
)
assert SPEC is not None
assert SPEC.loader is not None
watchdog = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = watchdog
SPEC.loader.exec_module(watchdog)


def snapshot(
    used_bytes: int,
    *,
    total_bytes: int = 200,
    active_swaps: tuple[str, ...] = (),
) -> Any:
    return watchdog.HostSnapshot(
        total_bytes=total_bytes,
        available_bytes=total_bytes - used_bytes,
        active_swaps=active_swaps,
    )


class SequenceReader:
    def __init__(self, values: list[Any]):
        self.values = values
        self.index = 0

    def read_snapshot(self) -> Any:
        index = min(self.index, len(self.values) - 1)
        self.index += 1
        value = self.values[index]
        if isinstance(value, Exception):
            raise value
        return value


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class FakeProcess:
    def __init__(self, returncode: int | None = None):
        self.pid = 4321
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout or 0.0)
        return self.returncode


class Harness:
    def __init__(
        self,
        values: list[Any],
        process: FakeProcess,
        signal_handler: Any | None = None,
    ):
        self.reader = SequenceReader(values)
        self.process = process
        self.signal_handler = signal_handler
        self.clock = FakeClock()
        self.stream = io.StringIO()
        self.launched = False
        self.signals: list[int] = []
        fixed_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.audit = watchdog.AuditLogger(
            self.stream, wall_clock=lambda: fixed_time
        )

    def launcher(self, command: tuple[str, ...], **kwargs: Any) -> FakeProcess:
        self.launched = True
        self.command = command
        self.launch_kwargs = kwargs
        return self.process

    def signal_group(self, process_group_id: int, signal_number: int) -> str:
        self.signals.append(signal_number)
        if self.signal_handler is not None:
            self.signal_handler(self.process, signal_number)
        return f"{signal.Signals(signal_number).name.lower()}_sent"

    def group_alive(self, process_group_id: int) -> bool:
        return self.process.returncode is None

    def run(self, **overrides: Any) -> int:
        config = watchdog.WatchdogConfig(
            command=("fake-command",),
            soft_bytes=100,
            emergency_bytes=150,
            grace_seconds=2,
            sample_interval_seconds=1,
            **overrides,
        )
        return watchdog.run_watchdog(
            config,
            reader=self.reader,
            audit=self.audit,
            launcher=self.launcher,
            signal_group=self.signal_group,
            group_alive=self.group_alive,
            monotonic=self.clock.monotonic,
            sleeper=self.clock.sleep,
        )

    def records(self) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in self.stream.getvalue().splitlines()
        ]


class TestProcfsParsing(unittest.TestCase):
    def test_parses_meminfo_as_integer_bytes_and_allows_zero_swap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "meminfo").write_text(
                "MemTotal:       131072 kB\n"
                "MemFree:          4096 kB\n"
                "MemAvailable:    32768 kB\n",
                encoding="utf-8",
            )
            (root / "swaps").write_text(
                "Filename Type Size Used Priority\n",
                encoding="utf-8",
            )

            result = watchdog.ProcfsReader(root).read_snapshot()

        self.assertEqual(result.total_bytes, 131072 * 1024)
        self.assertEqual(result.available_bytes, 32768 * 1024)
        self.assertEqual(result.used_bytes, 98304 * 1024)
        self.assertEqual(result.active_swaps, ())

    def test_rejects_active_swap_entry(self) -> None:
        content = (
            "Filename Type Size Used Priority\n"
            "/swapfile file 1048572 0 -2\n"
        )
        self.assertEqual(
            watchdog.ProcfsReader._parse_swaps(content),
            ("/swapfile",),
        )

    def test_rejects_malformed_or_missing_procfs_data(self) -> None:
        with self.assertRaisesRegex(
            watchdog.ProcfsError, "malformed MemAvailable"
        ):
            watchdog.ProcfsReader._parse_meminfo(
                "MemTotal: 10 kB\nMemAvailable: unknown\n"
            )
        with self.assertRaisesRegex(
            watchdog.ProcfsError, "missing MemAvailable"
        ):
            watchdog.ProcfsReader._parse_meminfo("MemTotal: 10 kB\n")
        with self.assertRaisesRegex(
            watchdog.ProcfsError, "malformed swaps header"
        ):
            watchdog.ProcfsReader._parse_swaps("")
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                watchdog.ProcfsError, "cannot read"
            ):
                watchdog.ProcfsReader(
                    Path(temp_dir)
                ).read_snapshot()


class TestWatchdogBehavior(unittest.TestCase):
    @staticmethod
    def _process_is_running(process_id: int) -> bool:
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(process_id)],
            capture_output=True,
            check=False,
            text=True,
        )
        return result.returncode == 0 and not result.stdout.lstrip().startswith(
            "Z"
        )

    @staticmethod
    def _write_procfs_fixture(root: Path) -> None:
        (root / "meminfo").write_text(
            "MemTotal: 131072 kB\nMemAvailable: 65536 kB\n",
            encoding="utf-8",
        )
        (root / "swaps").write_text(
            "Filename Type Size Used Priority\n",
            encoding="utf-8",
        )

    @staticmethod
    def _lease_arguments(root: Path) -> list[str]:
        return [
            "--lease-path",
            str(root / "lease.json"),
            "--heartbeat-path",
            str(root / "heartbeat.json"),
            "--audit-path",
            str(root / "persistent-audit.jsonl"),
        ]

    @staticmethod
    def _proc_stat(
        process_id: int,
        parent_id: int,
        process_group_id: int,
        start_time_ticks: int,
    ) -> str:
        fields = [
            "S",
            str(parent_id),
            str(process_group_id),
            *(["0"] * 16),
            str(start_time_ticks),
        ]
        return f"{process_id} (python) {' '.join(fields)}\n"

    def test_parent_signals_leave_no_child_or_grandchild(self) -> None:
        child_code = (
            "import os,signal,sys,time;"
            "signal.signal(signal.SIGHUP,signal.SIG_IGN);"
            "signal.signal(signal.SIGINT,signal.SIG_IGN);"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
            "grandchild=os.fork();"
            "\nif grandchild == 0:\n"
            " time.sleep(30)\n"
            "else:\n"
            " open(sys.argv[1],'w').write("
            "f'{os.getpid()} {grandchild}\\n');"
            " time.sleep(30)\n"
        )
        for signal_number in (
            signal.SIGHUP,
            signal.SIGINT,
            signal.SIGTERM,
        ):
            with self.subTest(signal=signal.Signals(signal_number).name):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    pid_file = root / "pids"
                    self._write_procfs_fixture(root)
                    stderr_path = root / "stderr.jsonl"
                    with stderr_path.open("w", encoding="utf-8") as audit:
                        wrapper = subprocess.Popen(
                            [
                                sys.executable,
                                str(SCRIPT_PATH),
                                "--procfs-root",
                                str(root),
                                *self._lease_arguments(root),
                                "--grace-seconds",
                                "0.2",
                                "--sample-interval-seconds",
                                "0.05",
                                "--",
                                sys.executable,
                                "-c",
                                child_code,
                                str(pid_file),
                            ],
                            stderr=audit,
                            text=True,
                        )
                        child_pid = None
                        grandchild_pid = None
                        try:
                            deadline = time.monotonic() + 5
                            while not pid_file.exists():
                                if time.monotonic() >= deadline:
                                    self.fail(
                                        "child process group did not start"
                                    )
                                time.sleep(0.01)
                            child_pid, grandchild_pid = (
                                int(value)
                                for value in pid_file.read_text(
                                    encoding="utf-8"
                                ).split()
                            )
                            time.sleep(0.05)
                            wrapper.send_signal(signal_number)
                            wrapper.wait(timeout=5)
                        finally:
                            if wrapper.poll() is None:
                                wrapper.kill()
                                wrapper.wait(timeout=5)
                            if child_pid is not None:
                                try:
                                    os.killpg(child_pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass

                    self.assertEqual(
                        wrapper.returncode, 128 + signal_number
                    )
                    records = [
                        json.loads(line)
                        for line in stderr_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                    ]
                    self.assertEqual(
                        records[-1]["classification"], "parent_signal"
                    )
                    signal_records = [
                        record
                        for record in records
                        if record["event"] == "process_group_signal"
                    ]
                    forwarded = [
                        record["signal"] for record in signal_records
                    ]
                    self.assertEqual(
                        forwarded[0],
                        signal.Signals(signal_number).name,
                    )
                    self.assertEqual(forwarded[-1], "SIGKILL")
                    self.assertEqual(
                        signal_records[0]["child_status"], "running"
                    )
                    self.assertIsNone(
                        signal_records[0]["child_returncode"]
                    )
                    self.assertLess(
                        signal_records[0]["timestamp"],
                        signal_records[-1]["timestamp"],
                    )
                    for process_id in (child_pid, grandchild_pid):
                        deadline = time.monotonic() + 2
                        while (
                            self._process_is_running(process_id)
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.01)
                        self.assertFalse(
                            self._process_is_running(process_id)
                        )
                    lease = json.loads(
                        (root / "lease.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    heartbeat = json.loads(
                        (root / "heartbeat.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    persistent_records = [
                        json.loads(line)
                        for line in (
                            root / "persistent-audit.jsonl"
                        ).read_text(encoding="utf-8").splitlines()
                    ]
                    self.assertEqual(lease["state"], "final")
                    self.assertEqual(
                        lease["final"]["classification"],
                        "parent_signal",
                    )
                    self.assertEqual(heartbeat["state"], "final")
                    self.assertEqual(
                        persistent_records[-1]["classification"],
                        "parent_signal",
                    )

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "Linux guardian lifecycle",
    )
    def test_parent_signal_grace_outlives_guardian_pulse_timeout(
        self,
    ) -> None:
        child_code = (
            "import os,signal,sys,time;"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
            "open(sys.argv[1],'w').write(str(os.getpid()));"
            "time.sleep(30)"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pid_file = root / "pid"
            stderr_path = root / "stderr.jsonl"
            self._write_procfs_fixture(root)
            with stderr_path.open("w", encoding="utf-8") as audit:
                wrapper = subprocess.Popen(
                    [
                        sys.executable,
                        str(SCRIPT_PATH),
                        "--procfs-root",
                        str(root),
                        *self._lease_arguments(root),
                        "--grace-seconds",
                        "0.4",
                        "--sample-interval-seconds",
                        "0.05",
                        "--heartbeat-max-age-seconds",
                        "0.1",
                        "--",
                        sys.executable,
                        "-c",
                        child_code,
                        str(pid_file),
                    ],
                    stderr=audit,
                    text=True,
                )
                child_pid = None
                try:
                    deadline = time.monotonic() + 5
                    while not pid_file.exists():
                        if time.monotonic() >= deadline:
                            self.fail("child process did not become ready")
                        time.sleep(0.01)
                    child_pid = int(
                        pid_file.read_text(encoding="utf-8")
                    )
                    started = time.monotonic()
                    wrapper.send_signal(signal.SIGTERM)
                    wrapper.wait(timeout=5)
                    elapsed = time.monotonic() - started
                finally:
                    if wrapper.poll() is None:
                        wrapper.kill()
                        wrapper.wait(timeout=5)
                    if child_pid is not None:
                        try:
                            os.killpg(child_pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

            records = [
                json.loads(line)
                for line in stderr_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            signals = [
                record["signal"]
                for record in records
                if record["event"] == "process_group_signal"
            ]
            self.assertGreaterEqual(elapsed, 0.35)
            self.assertEqual(
                wrapper.returncode,
                128 + signal.SIGTERM,
                records,
            )
            self.assertEqual(signals, ["SIGTERM", "SIGKILL"])
            self.assertEqual(
                records[-1]["classification"], "parent_signal"
            )
            self.assertFalse(self._process_is_running(child_pid))

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "Linux guardian lifecycle",
    )
    def test_parent_signal_allows_exit_after_pulse_deadline(self) -> None:
        child_code = (
            "import os,signal,sys,time\n"
            "def stop(_signal,_frame):\n"
            " time.sleep(0.25)\n"
            " raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM,stop)\n"
            "open(sys.argv[1],'w').write(str(os.getpid()))\n"
            "time.sleep(30)\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pid_file = root / "pid"
            stderr_path = root / "stderr.jsonl"
            self._write_procfs_fixture(root)
            with stderr_path.open("w", encoding="utf-8") as audit:
                wrapper = subprocess.Popen(
                    [
                        sys.executable,
                        str(SCRIPT_PATH),
                        "--procfs-root",
                        str(root),
                        *self._lease_arguments(root),
                        "--grace-seconds",
                        "0.4",
                        "--sample-interval-seconds",
                        "0.05",
                        "--heartbeat-max-age-seconds",
                        "0.1",
                        "--",
                        sys.executable,
                        "-c",
                        child_code,
                        str(pid_file),
                    ],
                    stderr=audit,
                    text=True,
                )
                try:
                    deadline = time.monotonic() + 5
                    while not pid_file.exists():
                        if time.monotonic() >= deadline:
                            self.fail("child process did not become ready")
                        time.sleep(0.01)
                    started = time.monotonic()
                    wrapper.send_signal(signal.SIGTERM)
                    wrapper.wait(timeout=5)
                    elapsed = time.monotonic() - started
                finally:
                    if wrapper.poll() is None:
                        wrapper.kill()
                        wrapper.wait(timeout=5)

            records = [
                json.loads(line)
                for line in stderr_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            signals = [
                record["signal"]
                for record in records
                if record["event"] == "process_group_signal"
            ]
            self.assertGreaterEqual(elapsed, 0.2)
            self.assertLess(elapsed, 0.4)
            self.assertEqual(wrapper.returncode, 128 + signal.SIGTERM)
            self.assertEqual(signals, ["SIGTERM"])
            self.assertEqual(records[-1]["child_returncode"], 0)

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "Linux guardian lifecycle",
    )
    def test_guardian_control_failure_still_kills_and_reaps_group(
        self,
    ) -> None:
        class FailingFinalAudit:
            def __init__(self, stream: Any, fail_at: int):
                self.stream = stream
                self.write_count = 0
                self.fail_at = fail_at

            def write(self, value: str) -> int:
                self.write_count += 1
                if self.write_count == self.fail_at:
                    raise OSError("audit write failed")
                return self.stream.write(value)

            def flush(self) -> None:
                self.stream.flush()

            def fileno(self) -> int:
                return self.stream.fileno()

            def close(self) -> None:
                self.stream.close()

        class FailingFinalLease:
            def finalize(self, record: dict[str, Any]) -> None:
                raise watchdog.ArtifactError(
                    "lease", "final lease write failed"
                )

        child_code = (
            "import os,signal,sys,time;"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
            "grandchild=os.fork();"
            "\nif grandchild == 0:\n"
            " time.sleep(30)\n"
            "else:\n"
            " open(sys.argv[1],'w').write("
            "f'{os.getpid()} {grandchild}\\n');"
            " time.sleep(30)\n"
        )
        for mode in ("closed", "blocked"):
            for artifact_failure in (
                "term_audit",
                "kill_audit",
                "final_audit",
                "lease",
            ):
                with self.subTest(
                    mode=mode,
                    artifact_failure=artifact_failure,
                ):
                    self._assert_guardian_control_failure_cleanup(
                        mode,
                        artifact_failure,
                        child_code,
                        FailingFinalAudit,
                        FailingFinalLease,
                    )

    def _assert_guardian_control_failure_cleanup(
        self,
        mode: str,
        artifact_failure: str,
        child_code: str,
        failing_final_audit: type,
        failing_final_lease: type,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pid_path = Path(temp_dir) / "pids"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    child_code,
                    str(pid_path),
                ],
                start_new_session=True,
            )
            read_fd, write_fd = os.pipe()
            os.set_blocking(write_fd, False)
            if mode == "closed":
                os.close(write_fd)
                write_fd = -1
            else:
                try:
                    while True:
                        os.write(write_fd, b"x" * 65536)
                except BlockingIOError:
                    pass
            guardian = watchdog.GuardianProcess(
                process,
                process.pid,
                write_fd,
            )
            stream = io.StringIO()
            audit = watchdog.AuditLogger(stream)
            if artifact_failure.endswith("_audit"):
                persistent_path = Path(temp_dir) / "persistent.jsonl"
                persistent_stream = persistent_path.open(
                    "w", encoding="utf-8"
                )
                audit.persistent_stream = failing_final_audit(
                    persistent_stream,
                    {
                        "term_audit": 1,
                        "kill_audit": 2,
                        "final_audit": 3,
                    }[artifact_failure],
                )
            else:
                audit.lease_manager = failing_final_lease()
            child_pid = None
            grandchild_pid = None
            try:
                deadline = time.monotonic() + 5
                while not pid_path.exists():
                    if time.monotonic() >= deadline:
                        self.fail("child process group did not start")
                    time.sleep(0.01)
                child_pid, grandchild_pid = (
                    int(value)
                    for value in pid_path.read_text(
                        encoding="utf-8"
                    ).split()
                )
                result = watchdog._graceful_cleanup(
                    audit,
                    guardian,
                    snapshot(50),
                    50,
                    "parent_signal",
                    128 + signal.SIGTERM,
                    "wrapper received SIGTERM",
                    signal.SIGTERM,
                    0.4,
                    watchdog._signal_process_group,
                    watchdog._process_group_alive,
                    time.monotonic,
                    time.sleep,
                )
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
                guardian.close()
                os.close(read_fd)

            records = [
                json.loads(line)
                for line in stream.getvalue().splitlines()
            ]
            signals = [
                record["signal"]
                for record in records
                if record["event"] == "process_group_signal"
            ]
            self.assertEqual(result, watchdog.EXIT_SIGNAL_ERROR)
            self.assertEqual(signals, ["SIGTERM", "SIGKILL"])
            self.assertEqual(
                records[-1]["classification"], "signal_error"
            )
            self.assertEqual(records[-1]["exit_code"], 7)
            self.assertEqual(
                records[-1]["threshold_reason"],
                "guardian control failed during graceful cleanup",
            )
            self.assertEqual(
                records[-1]["secondary_errors"][0]["component"],
                (
                    "audit"
                    if artifact_failure.endswith("_audit")
                    else "lease"
                ),
            )
            self.assertIn(
                (
                    "audit write failed"
                    if artifact_failure.endswith("_audit")
                    else "final lease write failed"
                ),
                records[-1]["secondary_errors"][0]["detail"],
            )
            self.assertEqual(
                records[-1]["child_returncode"],
                -signal.SIGKILL,
            )
            assert child_pid is not None
            assert grandchild_pid is not None
            for process_id in (child_pid, grandchild_pid):
                deadline = time.monotonic() + 2
                while (
                    self._process_is_running(process_id)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                self.assertFalse(self._process_is_running(process_id))

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "Linux guardian lifecycle",
    )
    def test_guardian_pipe_close_kills_group_without_fd_leak(self) -> None:
        child_code = (
            "import json,os,subprocess,sys,time\n"
            "targets=[]\n"
            "for name in os.listdir('/proc/self/fd'):\n"
            " try: targets.append(os.readlink('/proc/self/fd/'+name))\n"
            " except OSError: pass\n"
            "grandchild=subprocess.Popen([sys.executable,'-c',"
            "'import time;time.sleep(30)'])\n"
            "open(sys.argv[1],'w').write(json.dumps({"
            "'child':os.getpid(),'grandchild':grandchild.pid,"
            "'fds':targets}))\n"
            "time.sleep(30)\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state_path = root / "state.json"
            guardian = watchdog._launch_guardian(
                (
                    sys.executable,
                    "-c",
                    child_code,
                    str(state_path),
                ),
                os.environ.copy(),
                0.5,
                1.0,
                signal.pthread_sigmask(signal.SIG_BLOCK, ()),
            )
            control_target = os.readlink(
                f"/proc/self/fd/{guardian.pulse_fd}"
            )
            deadline = time.monotonic() + 5
            while not state_path.exists():
                if time.monotonic() >= deadline:
                    self.fail("guardian payload did not become ready")
                time.sleep(0.01)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            os.close(guardian.pulse_fd)
            guardian.wait(timeout=5)

        self.assertNotIn(control_target, state["fds"])
        for process_id in (state["child"], state["grandchild"]):
            deadline = time.monotonic() + 2
            while (
                self._process_is_running(process_id)
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            self.assertFalse(self._process_is_running(process_id))

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "Linux guardian lifecycle",
    )
    def test_guardian_documents_setsid_escape_limit(self) -> None:
        child_code = (
            "import os,subprocess,sys,time\n"
            "escaped=subprocess.Popen([sys.executable,'-c',"
            "'import time;time.sleep(30)'],start_new_session=True)\n"
            "open(sys.argv[1],'w').write(str(escaped.pid))\n"
            "time.sleep(30)\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "escaped-pid"
            guardian = watchdog._launch_guardian(
                (
                    sys.executable,
                    "-c",
                    child_code,
                    str(state_path),
                ),
                os.environ.copy(),
                0.5,
                1.0,
                signal.pthread_sigmask(signal.SIG_BLOCK, ()),
            )
            deadline = time.monotonic() + 5
            while not state_path.exists():
                if time.monotonic() >= deadline:
                    self.fail("escaped payload did not become ready")
                time.sleep(0.01)
            escaped_pid = int(
                state_path.read_text(encoding="utf-8")
            )
            os.close(guardian.pulse_fd)
            guardian.wait(timeout=5)
            self.assertTrue(self._process_is_running(escaped_pid))
            os.kill(escaped_pid, signal.SIGKILL)
            deadline = time.monotonic() + 2
            while (
                self._process_is_running(escaped_pid)
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            self.assertFalse(self._process_is_running(escaped_pid))

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "Linux guardian lifecycle",
    )
    def test_guard_kills_group_after_watchdog_loss_or_stall(self) -> None:
        child_code = (
            "import importlib.util,os,pathlib,subprocess,sys,time\n"
            "script=pathlib.Path(sys.argv[1])\n"
            "spec=importlib.util.spec_from_file_location('guard_watchdog',script)\n"
            "module=importlib.util.module_from_spec(spec)\n"
            "sys.modules[spec.name]=module\n"
            "spec.loader.exec_module(module)\n"
            "module.start_process_group_lease_guard("
            "script,expected_procfs_root=pathlib.Path(sys.argv[2]))\n"
            "grandchild=subprocess.Popen([sys.executable,'-c',"
            "'import time;time.sleep(30)'])\n"
            "open(sys.argv[3],'w').write("
            "f'{os.getpid()} {grandchild.pid}\\n')\n"
            "time.sleep(30)\n"
        )
        for mode in ("sigkill", "sigstop"):
            with self.subTest(mode=mode):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    pid_path = root / "pids"
                    self._write_procfs_fixture(root)
                    wrapper = subprocess.Popen(
                        [
                            sys.executable,
                            str(SCRIPT_PATH),
                            "--procfs-root",
                            str(root),
                            *self._lease_arguments(root),
                            "--heartbeat-max-age-seconds",
                            "0.3",
                            "--sample-interval-seconds",
                            "0.05",
                            "--",
                            sys.executable,
                            "-c",
                            child_code,
                            str(SCRIPT_PATH),
                            str(root),
                            str(pid_path),
                        ],
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    deadline = time.monotonic() + 5
                    while not pid_path.exists():
                        if wrapper.poll() is not None:
                            assert wrapper.stderr is not None
                            self.fail(wrapper.stderr.read())
                        if time.monotonic() >= deadline:
                            self.fail(
                                "guarded payload did not become ready"
                            )
                        time.sleep(0.01)
                    child_pid, grandchild_pid = (
                        int(value)
                        for value in pid_path.read_text(
                            encoding="utf-8"
                        ).split()
                    )
                    if mode == "sigkill":
                        wrapper.kill()
                    else:
                        os.kill(wrapper.pid, signal.SIGSTOP)
                        heartbeat_path = root / "heartbeat.json"
                        heartbeat = json.loads(
                            heartbeat_path.read_text(encoding="utf-8")
                        )
                        heartbeat["updated_monotonic_ns"] = (
                            time.monotonic_ns()
                        )
                        watchdog._write_json_atomic(
                            heartbeat_path, heartbeat
                        )
                        time.sleep(0.7)
                        os.kill(wrapper.pid, signal.SIGCONT)
                    wrapper.wait(timeout=5)
                    for process_id in (child_pid, grandchild_pid):
                        deadline = time.monotonic() + 2
                        while (
                            self._process_is_running(process_id)
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.01)
                        self.assertFalse(
                            self._process_is_running(process_id)
                        )
                    if wrapper.stderr is not None:
                        wrapper.stderr.close()

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "Linux guardian lifecycle",
    )
    def test_payload_guard_fails_closed_on_artifact_error(self) -> None:
        child_code = (
            "import importlib.util,os,pathlib,subprocess,sys,time\n"
            "script=pathlib.Path(sys.argv[1])\n"
            "spec=importlib.util.spec_from_file_location('guard_watchdog',script)\n"
            "module=importlib.util.module_from_spec(spec)\n"
            "sys.modules[spec.name]=module\n"
            "spec.loader.exec_module(module)\n"
            "module.start_process_group_lease_guard("
            "script,expected_procfs_root=pathlib.Path(sys.argv[2]))\n"
            "def fail(*_args,**_kwargs):\n"
            " raise module.ArtifactError('script','unreadable')\n"
            "module.validate_active_lease=fail\n"
            "grandchild=subprocess.Popen([sys.executable,'-c',"
            "'import time;time.sleep(30)'])\n"
            "open(sys.argv[3],'w').write("
            "f'{os.getpid()} {grandchild.pid}\\n')\n"
            "time.sleep(30)\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pid_path = root / "pids"
            self._write_procfs_fixture(root)
            wrapper = subprocess.Popen(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--procfs-root",
                    str(root),
                    *self._lease_arguments(root),
                    "--heartbeat-max-age-seconds",
                    "0.3",
                    "--sample-interval-seconds",
                    "0.05",
                    "--",
                    sys.executable,
                    "-c",
                    child_code,
                    str(SCRIPT_PATH),
                    str(root),
                    str(pid_path),
                ],
                stderr=subprocess.PIPE,
                text=True,
            )
            child_pid = None
            grandchild_pid = None
            try:
                deadline = time.monotonic() + 5
                while not pid_path.exists():
                    if wrapper.poll() is not None:
                        assert wrapper.stderr is not None
                        self.fail(wrapper.stderr.read())
                    if time.monotonic() >= deadline:
                        self.fail("guarded payload did not become ready")
                    time.sleep(0.01)
                child_pid, grandchild_pid = (
                    int(value)
                    for value in pid_path.read_text(
                        encoding="utf-8"
                    ).split()
                )
                wrapper.wait(timeout=5)
            finally:
                if wrapper.poll() is None:
                    wrapper.kill()
                    wrapper.wait(timeout=5)
                if wrapper.stderr is not None:
                    wrapper.stderr.close()
                if child_pid is not None:
                    try:
                        os.killpg(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

            assert child_pid is not None
            assert grandchild_pid is not None
            for process_id in (child_pid, grandchild_pid):
                deadline = time.monotonic() + 2
                while (
                    self._process_is_running(process_id)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                self.assertFalse(self._process_is_running(process_id))

    def test_child_sigterm_handler_exits_without_escalation(self) -> None:
        child_code = (
            "import os,signal,sys,time\n"
            "def stop(_signal,_frame):\n"
            " open(sys.argv[2],'w').write('handled\\n')\n"
            " raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM,stop)\n"
            "open(sys.argv[1],'w').write(f'{os.getpid()}\\n')\n"
            "time.sleep(30)\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ready_path = root / "ready"
            handled_path = root / "handled"
            audit_path = root / "audit.jsonl"
            self._write_procfs_fixture(root)
            with audit_path.open("w", encoding="utf-8") as audit:
                wrapper = subprocess.Popen(
                    [
                        sys.executable,
                        str(SCRIPT_PATH),
                        "--procfs-root",
                        str(root),
                        "--grace-seconds",
                        "0.5",
                        "--sample-interval-seconds",
                        "0.05",
                        "--",
                        sys.executable,
                        "-c",
                        child_code,
                        str(ready_path),
                        str(handled_path),
                    ],
                    stderr=audit,
                    text=True,
                )
                deadline = time.monotonic() + 5
                while not ready_path.exists():
                    if time.monotonic() >= deadline:
                        wrapper.kill()
                        wrapper.wait(timeout=5)
                        self.fail("SIGTERM child did not become ready")
                    time.sleep(0.01)
                child_pid = int(
                    ready_path.read_text(encoding="utf-8").strip()
                )
                try:
                    wrapper.send_signal(signal.SIGTERM)
                    wrapper.wait(timeout=5)
                finally:
                    if wrapper.poll() is None:
                        wrapper.kill()
                        wrapper.wait(timeout=5)
                    try:
                        os.killpg(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

            records = [
                json.loads(line)
                for line in audit_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            forwarded = [
                record["signal"]
                for record in records
                if record["event"] == "process_group_signal"
            ]
            self.assertEqual(wrapper.returncode, 128 + signal.SIGTERM)
            self.assertTrue(handled_path.exists())
            self.assertEqual(forwarded, ["SIGTERM"])
            self.assertEqual(records[-1]["child_returncode"], 0)

    def test_leader_exit_cleans_up_surviving_grandchild(self) -> None:
        child_code = (
            "import os,signal,sys,time;"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
            "grandchild=os.fork();"
            "\nif grandchild == 0:\n"
            " time.sleep(30)\n"
            "else:\n"
            " open(sys.argv[1],'w').write("
            "f'{os.getpid()} {grandchild}\\n')\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pid_file = root / "pids"
            audit_path = root / "audit.jsonl"
            self._write_procfs_fixture(root)
            with audit_path.open("w", encoding="utf-8") as audit:
                wrapper = subprocess.Popen(
                    [
                        sys.executable,
                        str(SCRIPT_PATH),
                        "--procfs-root",
                        str(root),
                        "--grace-seconds",
                        "0.2",
                        "--sample-interval-seconds",
                        "0.05",
                        "--",
                        sys.executable,
                        "-c",
                        child_code,
                        str(pid_file),
                    ],
                    stderr=audit,
                    text=True,
                )
                deadline = time.monotonic() + 5
                while not pid_file.exists():
                    if time.monotonic() >= deadline:
                        wrapper.kill()
                        wrapper.wait(timeout=5)
                        self.fail("leader process did not write child PIDs")
                    time.sleep(0.01)
                child_pid, grandchild_pid = (
                    int(value)
                    for value in pid_file.read_text(
                        encoding="utf-8"
                    ).split()
                )
                try:
                    wrapper.wait(timeout=5)
                finally:
                    if wrapper.poll() is None:
                        wrapper.kill()
                        wrapper.wait(timeout=5)
                    try:
                        os.killpg(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

            records = [
                json.loads(line)
                for line in audit_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            forwarded = [
                record["signal"]
                for record in records
                if record["event"] == "process_group_signal"
            ]
            self.assertEqual(wrapper.returncode, 0)
            self.assertEqual(records[-1]["classification"], "child_exit")
            self.assertEqual(records[-1]["child_returncode"], 0)
            self.assertEqual(forwarded, ["SIGTERM", "SIGKILL"])
            for process_id in (child_pid, grandchild_pid):
                deadline = time.monotonic() + 2
                while (
                    self._process_is_running(process_id)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                self.assertFalse(self._process_is_running(process_id))

    def test_soft_limit_descendant_escalation_is_grace_timeout(self) -> None:
        child_code = (
            "import os,signal,sys,time\n"
            "def stop(_signal,_frame):\n"
            " raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM,stop)\n"
            "grandchild=os.fork()\n"
            "if grandchild == 0:\n"
            " signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            " time.sleep(30)\n"
            "else:\n"
            " open(sys.argv[1],'w').write("
            "f'{os.getpid()} {grandchild}\\n')\n"
            " time.sleep(30)\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pid_file = root / "pids"
            audit_path = root / "audit.jsonl"
            (root / "meminfo").write_text(
                "MemTotal: 3145728 kB\n"
                "MemAvailable: 2621440 kB\n",
                encoding="utf-8",
            )
            (root / "swaps").write_text(
                "Filename Type Size Used Priority\n",
                encoding="utf-8",
            )
            with audit_path.open("w", encoding="utf-8") as audit:
                wrapper = subprocess.Popen(
                    [
                        sys.executable,
                        str(SCRIPT_PATH),
                        "--procfs-root",
                        str(root),
                        "--soft-gib",
                        "1",
                        "--emergency-gib",
                        "2",
                        "--grace-seconds",
                        "0.2",
                        "--sample-interval-seconds",
                        "0.05",
                        "--",
                        sys.executable,
                        "-c",
                        child_code,
                        str(pid_file),
                    ],
                    stderr=audit,
                    text=True,
                )
                deadline = time.monotonic() + 5
                while not pid_file.exists():
                    if time.monotonic() >= deadline:
                        wrapper.kill()
                        wrapper.wait(timeout=5)
                        self.fail("soft-limit process group did not start")
                    time.sleep(0.01)
                child_pid, grandchild_pid = (
                    int(value)
                    for value in pid_file.read_text(
                        encoding="utf-8"
                    ).split()
                )
                next_meminfo = root / "meminfo.next"
                next_meminfo.write_text(
                    "MemTotal: 3145728 kB\n"
                    "MemAvailable: 1572864 kB\n",
                    encoding="utf-8",
                )
                next_meminfo.replace(root / "meminfo")
                try:
                    wrapper.wait(timeout=5)
                finally:
                    if wrapper.poll() is None:
                        wrapper.kill()
                        wrapper.wait(timeout=5)
                    try:
                        os.killpg(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

            records = [
                json.loads(line)
                for line in audit_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            forwarded = [
                record["signal"]
                for record in records
                if record["event"] == "process_group_signal"
            ]
            final = records[-1]
            self.assertEqual(wrapper.returncode, watchdog.EXIT_GRACE_TIMEOUT)
            self.assertEqual(final["classification"], "grace_timeout")
            self.assertEqual(final["child_returncode"], 0)
            self.assertEqual(forwarded, ["SIGTERM", "SIGKILL"])
            for process_id in (child_pid, grandchild_pid):
                deadline = time.monotonic() + 2
                while (
                    self._process_is_running(process_id)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                self.assertFalse(self._process_is_running(process_id))

    def test_configuration_rejects_non_finite_timing(self) -> None:
        config = watchdog.WatchdogConfig(
            command=("fake-command",),
            grace_seconds=float("nan"),
        )
        with self.assertRaisesRegex(ValueError, "grace period"):
            config.validate()

    def test_configuration_rejects_weakened_liveness_timing(self) -> None:
        cases = (
            (
                {"grace_seconds": 31.0},
                "grace period",
            ),
            (
                {"sample_interval_seconds": 1.1},
                "sample interval",
            ),
            (
                {
                    "sample_interval_seconds": 1.0,
                    "heartbeat_max_age_seconds": 5.1,
                },
                "heartbeat max age",
            ),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                config = watchdog.WatchdogConfig(
                    command=("fake-command",),
                    **overrides,
                )
                with self.assertRaisesRegex(ValueError, message):
                    config.validate()

    def test_stderr_failure_does_not_bypass_cleanup(self) -> None:
        class FailingStderr(io.StringIO):
            def write(self, value: str) -> int:
                raise OSError("stderr closed")

        process = FakeProcess()

        def exit_on_kill(
            target: FakeProcess, signal_number: int
        ) -> None:
            if signal_number == signal.SIGKILL:
                target.returncode = -signal.SIGKILL

        harness = Harness(
            [snapshot(50)],
            process,
            signal_handler=exit_on_kill,
        )
        harness.audit = watchdog.AuditLogger(FailingStderr())
        with tempfile.TemporaryDirectory() as temp_dir:
            persistent_path = Path(temp_dir) / "audit.jsonl"
            harness.audit.open_persistent(persistent_path)
            result = watchdog._graceful_cleanup(
                harness.audit,
                process,
                snapshot(50),
                50,
                "internal_error",
                watchdog.EXIT_INTERNAL_ERROR,
                "test cleanup",
                signal.SIGTERM,
                0.1,
                harness.signal_group,
                harness.group_alive,
                harness.clock.monotonic,
                harness.clock.sleep,
            )
            harness.audit.close()
            records = [
                json.loads(line)
                for line in persistent_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        self.assertEqual(
            harness.signals, [signal.SIGTERM, signal.SIGKILL]
        )
        self.assertEqual(result, watchdog.EXIT_LEASE_ERROR)
        self.assertEqual(records[-1]["classification"], "lease_error")

    def test_audit_write_and_close_failures_do_not_bypass_cleanup(
        self,
    ) -> None:
        class FailingPersistent(io.StringIO):
            def __init__(self) -> None:
                super().__init__()
                self.close_called = False

            def write(self, value: str) -> int:
                raise OSError("persistent write failed")

            def close(self) -> None:
                if self.close_called:
                    super().close()
                    return
                self.close_called = True
                raise OSError("persistent close failed")

        process = FakeProcess()

        def exit_on_kill(
            target: FakeProcess, signal_number: int
        ) -> None:
            if signal_number == signal.SIGKILL:
                target.returncode = -signal.SIGKILL

        harness = Harness(
            [snapshot(50)],
            process,
            signal_handler=exit_on_kill,
        )
        persistent = FailingPersistent()
        harness.audit.persistent_stream = persistent
        result = watchdog._graceful_cleanup(
            harness.audit,
            process,
            snapshot(50),
            50,
            "internal_error",
            watchdog.EXIT_INTERNAL_ERROR,
            "test cleanup",
            signal.SIGTERM,
            0.1,
            harness.signal_group,
            harness.group_alive,
            harness.clock.monotonic,
            harness.clock.sleep,
        )

        self.assertTrue(persistent.close_called)
        self.assertEqual(
            harness.signals, [signal.SIGTERM, signal.SIGKILL]
        )
        self.assertEqual(process.returncode, -signal.SIGKILL)
        self.assertEqual(result, watchdog.EXIT_LEASE_ERROR)
        self.assertEqual(
            harness.records()[-1]["classification"], "lease_error"
        )

    def test_final_record_survives_artifact_failures(self) -> None:
        class FailingLease:
            def finalize(self, record: dict[str, Any]) -> None:
                raise watchdog.ArtifactError("lease", "write failed")

        class FailingAudit(io.StringIO):
            def write(self, value: str) -> int:
                raise OSError("write failed")

        for component in ("lease", "audit"):
            with self.subTest(component=component):
                stream = io.StringIO()
                audit = watchdog.AuditLogger(stream)
                if component == "lease":
                    setattr(audit, "lease_manager", FailingLease())
                else:
                    audit.persistent_stream = FailingAudit()
                result = watchdog._emit_final(
                    audit,
                    "child_exit",
                    0,
                    "child exited",
                    snapshot(50),
                    50,
                )
                records = [
                    json.loads(line)
                    for line in stream.getvalue().splitlines()
                ]
                self.assertEqual(result, watchdog.EXIT_LEASE_ERROR)
                self.assertEqual(
                    records[-1]["classification"], "lease_error"
                )
                self.assertTrue(audit.finalized)
                self.assertEqual(
                    audit.final_exit_code, watchdog.EXIT_LEASE_ERROR
                )

    def test_emergency_signal_precedes_artifact_write(self) -> None:
        events: list[str] = []

        class BlockingAudit(watchdog.AuditLogger):
            def __init__(self) -> None:
                super().__init__(io.StringIO())
                self.calls = 0

            def emit(
                self, event: str, **fields: object
            ) -> dict[str, object]:
                self.calls += 1
                events.append(f"audit:{event}")
                if self.calls == 3:
                    raise watchdog.ArtifactError(
                        "audit", "simulated blocked fsync"
                    )
                return super().emit(event, **fields)

        def exit_on_kill(
            process: FakeProcess, signal_number: int
        ) -> None:
            events.append(f"signal:{signal_number}")
            if signal_number == signal.SIGKILL:
                process.returncode = -signal.SIGKILL

        harness = Harness(
            [snapshot(50), snapshot(160)],
            FakeProcess(),
            signal_handler=exit_on_kill,
        )
        harness.audit = BlockingAudit()

        result = harness.run()

        self.assertEqual(result, watchdog.EXIT_LEASE_ERROR)
        self.assertEqual(events[2], f"signal:{signal.SIGKILL}")
        self.assertEqual(events[3], "audit:process_group_signal")
        self.assertEqual(harness.process.returncode, -signal.SIGKILL)

    def test_cleanup_reaps_after_persistent_audit_failure(self) -> None:
        class FailingSignalAudit(watchdog.AuditLogger):
            def emit(
                self, event: str, **fields: object
            ) -> dict[str, object]:
                if event == "process_group_signal":
                    raise watchdog.ArtifactError(
                        "audit", "simulated persistent write failure"
                    )
                return super().emit(event, **fields)

        process = FakeProcess()
        signals: list[int] = []
        clock = FakeClock()

        def signal_group(
            process_group_id: int, signal_number: int
        ) -> str:
            signals.append(signal_number)
            if signal_number == signal.SIGKILL:
                process.returncode = -signal.SIGKILL
            return f"{signal.Signals(signal_number).name.lower()}_sent"

        result = watchdog._graceful_cleanup(
            FailingSignalAudit(io.StringIO()),
            process,
            snapshot(50),
            50,
            "parent_signal",
            128 + signal.SIGTERM,
            "wrapper received SIGTERM",
            signal.SIGTERM,
            0.1,
            signal_group,
            lambda _process_group_id: process.returncode is None,
            clock.monotonic,
            clock.sleep,
        )

        self.assertEqual(result, watchdog.EXIT_LEASE_ERROR)
        self.assertEqual(signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(process.returncode, -signal.SIGKILL)

    def test_invalid_artifact_path_emits_configuration_final(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--lease-path",
                "~strix-watchdog-user-does-not-exist/lease.json",
                "--heartbeat-path",
                "/tmp/heartbeat.json",
                "--audit-path",
                "/tmp/audit.jsonl",
                "--",
                sys.executable,
                "-c",
                "pass",
            ],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(result.returncode, watchdog.EXIT_PROCFS_ERROR)
        final = json.loads(result.stderr.splitlines()[-1])
        self.assertEqual(final["classification"], "configuration_error")

    def test_cli_fixture_launches_command_and_propagates_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            child_result_path = root / "child-result.json"
            self._write_procfs_fixture(root)
            child_code = (
                "import json,os,sys,time\n"
                "keys=('STRIX_MEMORY_WATCHDOG_LEASE_PATH',"
                "'STRIX_MEMORY_WATCHDOG_HEARTBEAT_PATH',"
                "'STRIX_MEMORY_WATCHDOG_AUDIT_PATH')\n"
                "deadline=time.monotonic()+5\n"
                "while True:\n"
                " try:\n"
                "  with open(os.environ[keys[0]],encoding='utf-8') as stream:\n"
                "   lease=json.load(stream)\n"
                "  break\n"
                " except (OSError,json.JSONDecodeError):\n"
                "  if time.monotonic()>=deadline: raise\n"
                "  time.sleep(0.01)\n"
                "assert os.getpgrp()==lease['child_process_group_id']\n"
                "open(sys.argv[1],'w').write(json.dumps({"
                "key:os.environ[key] for key in keys}))\n"
                "raise SystemExit(23)\n"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--procfs-root",
                    str(root),
                    *self._lease_arguments(root),
                    "--sample-interval-seconds",
                    "0.01",
                    "--",
                    sys.executable,
                    "-c",
                    child_code,
                    str(child_result_path),
                ],
                capture_output=True,
                check=False,
                text=True,
            )
            lease = json.loads(
                (root / "lease.json").read_text(encoding="utf-8")
            )
            persistent_records = [
                json.loads(line)
                for line in (
                    root / "persistent-audit.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            child_result = json.loads(
                child_result_path.read_text(encoding="utf-8")
            )

        self.assertEqual(result.returncode, 23)
        records = [
            json.loads(line) for line in result.stderr.splitlines()
        ]
        self.assertEqual(records[-1]["classification"], "child_exit")
        self.assertEqual(records[-1]["child_returncode"], 23)
        self.assertEqual(lease["format"], watchdog.LEASE_FORMAT)
        self.assertEqual(lease["version"], watchdog.LEASE_VERSION)
        self.assertEqual(lease["state"], "final")
        self.assertEqual(lease["soft_bytes"], 116 * 1024**3)
        self.assertEqual(
            lease["emergency_bytes"], 118 * 1024**3
        )
        self.assertEqual(
            lease["child_command_sha256"],
            watchdog._command_sha256(
                (
                    sys.executable,
                    "-c",
                    child_code,
                    str(child_result_path),
                )
            ),
        )
        self.assertEqual(
            lease["final"]["classification"], "child_exit"
        )
        self.assertEqual(
            persistent_records[-1]["classification"], "child_exit"
        )
        self.assertEqual(
            child_result["STRIX_MEMORY_WATCHDOG_LEASE_PATH"],
            str((root / "lease.json").resolve()),
        )
        self.assertEqual(
            child_result["STRIX_MEMORY_WATCHDOG_HEARTBEAT_PATH"],
            str((root / "heartbeat.json").resolve()),
        )
        self.assertEqual(
            child_result["STRIX_MEMORY_WATCHDOG_AUDIT_PATH"],
            str((root / "persistent-audit.jsonl").resolve()),
        )

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "Linux guardian lifecycle",
    )
    def test_guardian_preserves_payload_signal_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_procfs_fixture(root)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--procfs-root",
                    str(root),
                    *self._lease_arguments(root),
                    "--sample-interval-seconds",
                    "0.01",
                    "--",
                    sys.executable,
                    "-c",
                    (
                        "import os,signal,time;"
                        "time.sleep(0.1);"
                        "os.kill(os.getpid(),signal.SIGTERM)"
                    ),
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
            records = [
                json.loads(line)
                for line in result.stderr.splitlines()
            ]

        self.assertEqual(result.returncode, 128 + signal.SIGTERM)
        self.assertEqual(records[-1]["classification"], "child_exit")
        self.assertEqual(
            records[-1]["child_returncode"], -signal.SIGTERM
        )
        self.assertEqual(records[-1]["child_status"], "signaled")

    def test_existing_lease_fails_closed_and_stops_child(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_procfs_fixture(root)
            lease_path = root / "lease.json"
            lease_path.write_text("untrusted\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--procfs-root",
                    str(root),
                    *self._lease_arguments(root),
                    "--grace-seconds",
                    "0.1",
                    "--sample-interval-seconds",
                    "0.05",
                    "--",
                    sys.executable,
                    "-c",
                    "import time;time.sleep(30)",
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
            records = [
                json.loads(line) for line in result.stderr.splitlines()
            ]
            child_pid = records[-1]["child_pid"]
            self.assertEqual(
                lease_path.read_text(encoding="utf-8"), "untrusted\n"
            )

        self.assertEqual(result.returncode, watchdog.EXIT_LEASE_ERROR)
        self.assertEqual(records[-1]["classification"], "lease_error")
        self.assertFalse(self._process_is_running(child_pid))

    def test_active_lease_validation_rejects_tamper_and_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            process_root = root / "proc"
            watchdog_pid = 1200
            guardian_pid = 1250
            child_pid = 1300
            current_pid = 1400
            watchdog_start_ticks = 456789
            script_path = root / "watchdog.py"
            script_path.write_text("print('watchdog')\n", encoding="utf-8")
            lease_path = root / "lease.json"
            heartbeat_path = root / "heartbeat.json"
            audit_path = root / "audit.jsonl"
            command = [sys.executable, "run_matrix.py"]
            argv = [
                sys.executable,
                "watchdog.py",
                "--procfs-root",
                "/proc",
                "--lease-path",
                str(lease_path),
                "--heartbeat-path",
                str(heartbeat_path),
                "--audit-path",
                str(audit_path),
                "--",
                *command,
            ]
            cmdline = b"\0".join(os.fsencode(value) for value in argv)
            for process_id, parent_id, group_id, start_ticks in (
                (watchdog_pid, 1, watchdog_pid, watchdog_start_ticks),
                (guardian_pid, watchdog_pid, guardian_pid, 456790),
                (child_pid, guardian_pid, guardian_pid, 456791),
                (current_pid, child_pid, guardian_pid, 456792),
            ):
                process_dir = process_root / str(process_id)
                process_dir.mkdir(parents=True)
                (process_dir / "stat").write_text(
                    self._proc_stat(
                        process_id,
                        parent_id,
                        group_id,
                        start_ticks,
                    ),
                    encoding="utf-8",
                )
            (process_root / str(watchdog_pid) / "cwd").symlink_to(
                root, target_is_directory=True
            )
            (process_root / str(watchdog_pid) / "exe").symlink_to(
                Path(sys.executable).resolve()
            )
            (process_root / str(watchdog_pid) / "cmdline").write_bytes(
                cmdline
            )

            audit_line = (
                '{"event":"child_started",'
                '"timestamp":"2026-01-01T00:00:00Z"}\n'
            )
            audit_descriptor = os.open(
                audit_path,
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
                0o600,
            )
            os.write(audit_descriptor, audit_line.encode("utf-8"))
            os.fsync(audit_descriptor)
            fcntl.flock(
                audit_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB
            )
            audit_status = os.fstat(audit_descriptor)
            fd_root = process_root / str(watchdog_pid) / "fd"
            fd_root.mkdir()
            (fd_root / "9").symlink_to(audit_path)
            lease = {
                "format": watchdog.LEASE_FORMAT,
                "version": watchdog.LEASE_VERSION,
                "lease_id": "test-lease",
                "state": "active",
                "watchdog_pid": watchdog_pid,
                "watchdog_start_time_utc": "2026-01-01T00:00:00.000Z",
                "watchdog_start_time_ticks": watchdog_start_ticks,
                "watchdog_command_sha256": hashlib.sha256(
                    cmdline
                ).hexdigest(),
                "watchdog_executable_path": str(
                    Path(sys.executable).resolve()
                ),
                "watchdog_script_path": str(script_path),
                "watchdog_script_sha256": hashlib.sha256(
                    script_path.read_bytes()
                ).hexdigest(),
                "soft_bytes": watchdog.DEFAULT_SOFT_BYTES,
                "emergency_bytes": watchdog.DEFAULT_EMERGENCY_BYTES,
                "strict_ceiling_bytes": watchdog.STRICT_CEILING_BYTES,
                "grace_seconds": watchdog.DEFAULT_GRACE_SECONDS,
                "sample_interval_seconds": (
                    watchdog.DEFAULT_SAMPLE_INTERVAL_SECONDS
                ),
                "guardian_pid": guardian_pid,
                "child_pid": child_pid,
                "child_process_group_id": guardian_pid,
                "command": command,
                "child_command_sha256": watchdog._command_sha256(
                    command
                ),
                "heartbeat_path": str(heartbeat_path),
                "max_heartbeat_age_seconds": 5.0,
                "audit_path": str(audit_path),
                "audit_device": audit_status.st_dev,
                "audit_inode": audit_status.st_ino,
                "audit_uid": audit_status.st_uid,
                "audit_mode": 0o600,
                "audit_fd": 9,
                "procfs_root": "/proc",
            }
            heartbeat = {
                "format": watchdog.HEARTBEAT_FORMAT,
                "version": watchdog.HEARTBEAT_VERSION,
                "lease_id": "test-lease",
                "sequence": 4,
                "state": "active",
                "updated_at": "2026-01-01T00:00:01.000Z",
                "updated_monotonic_ns": 9_000_000_000,
                "watchdog_pid": watchdog_pid,
                "watchdog_start_time_ticks": (
                    watchdog_start_ticks
                ),
                "child_pid": child_pid,
                "child_process_group_id": guardian_pid,
                "sample": {
                    "audit_record_sha256": hashlib.sha256(
                        audit_line.encode("utf-8")
                    ).hexdigest()
                },
            }
            watchdog._write_json_atomic(
                lease_path, lease, create=True
            )
            watchdog._write_json_atomic(
                heartbeat_path, heartbeat, create=True
            )

            try:
                validation_args = {
                    "expected_script_path": script_path,
                    "expected_executable_path": Path(sys.executable),
                    "expected_command": command,
                    "expected_heartbeat_path": heartbeat_path,
                    "expected_audit_path": audit_path,
                    "expected_max_heartbeat_age_seconds": 5.0,
                    "current_process_id": current_pid,
                    "process_procfs_root": process_root,
                    "monotonic_ns": lambda: 10_000_000_000,
                    "pidfd_open": lambda _pid: os.open(
                        os.devnull, os.O_RDONLY
                    ),
                }
                validated = watchdog.validate_active_lease(
                    lease_path, **validation_args
                )
                self.assertEqual(validated["lease_id"], "test-lease")

                def publish_lease(
                    value: dict[str, Any],
                    process_argv: list[str] = argv,
                ) -> None:
                    process_cmdline = b"\0".join(
                        os.fsencode(argument)
                        for argument in process_argv
                    )
                    (process_root / str(watchdog_pid) / "cmdline").write_bytes(
                        process_cmdline
                    )
                    value["watchdog_command_sha256"] = hashlib.sha256(
                        process_cmdline
                    ).hexdigest()
                    watchdog._write_json_atomic(lease_path, value)

                for name, bad_argv in (
                    (
                        "helper inert argument",
                        [
                            sys.executable,
                            "helper.py",
                            str(script_path),
                            *argv[2:],
                        ],
                    ),
                    (
                        "python command string",
                        [
                            sys.executable,
                            "-c",
                            "pass",
                            str(script_path),
                            *argv[2:],
                        ],
                    ),
                    (
                        "python module",
                        [
                            sys.executable,
                            "-m",
                            "helper",
                            str(script_path),
                            *argv[2:],
                        ],
                    ),
                    (
                        "interpreter option before script",
                        [
                            sys.executable,
                            "-O",
                            str(script_path),
                            *argv[2:],
                        ],
                    ),
                ):
                    with self.subTest(name):
                        publish_lease(dict(lease), bad_argv)
                        with self.assertRaisesRegex(
                            watchdog.LeaseValidationError,
                            "executable argv position",
                        ):
                            watchdog.validate_active_lease(
                                lease_path, **validation_args
                            )

                with self.subTest("wrong command-line policy"):
                    bad_argv = list(argv)
                    procfs_index = bad_argv.index("--procfs-root") + 1
                    bad_argv[procfs_index] = "/tmp/not-proc"
                    publish_lease(dict(lease), bad_argv)
                    with self.assertRaisesRegex(
                        watchdog.LeaseValidationError,
                        "command-line policy",
                    ):
                        watchdog.validate_active_lease(
                            lease_path, **validation_args
                        )

                with self.subTest("wrong lease timing policy"):
                    bad_lease = dict(lease)
                    bad_lease["grace_seconds"] = 29.0
                    publish_lease(bad_lease)
                    with self.assertRaisesRegex(
                        watchdog.LeaseValidationError,
                        "lease timing policy",
                    ):
                        watchdog.validate_active_lease(
                            lease_path, **validation_args
                        )

                with self.subTest("wrong monitored command"):
                    bad_argv = [*argv[:-1], "other_matrix.py"]
                    bad_lease = dict(lease)
                    bad_lease["command"] = [
                        sys.executable,
                        "other_matrix.py",
                    ]
                    bad_lease["child_command_sha256"] = (
                        watchdog._command_sha256(
                            bad_lease["command"]
                        )
                    )
                    publish_lease(bad_lease, bad_argv)
                    with self.assertRaisesRegex(
                        watchdog.LeaseValidationError,
                        "monitored command",
                    ):
                        watchdog.validate_active_lease(
                            lease_path, **validation_args
                        )

                with self.subTest("tampered script SHA"):
                    tampered = dict(lease)
                    tampered["watchdog_script_sha256"] = "0" * 64
                    publish_lease(tampered)
                    with self.assertRaisesRegex(
                        watchdog.LeaseValidationError, "script SHA"
                    ):
                        watchdog.validate_active_lease(
                            lease_path, **validation_args
                        )

                with self.subTest("stale heartbeat"):
                    publish_lease(dict(lease))
                    heartbeat["updated_monotonic_ns"] = 1
                    watchdog._write_json_atomic(
                        heartbeat_path, heartbeat
                    )
                    with self.assertRaisesRegex(
                        watchdog.LeaseValidationError,
                        "heartbeat is stale",
                    ):
                        watchdog.validate_active_lease(
                            lease_path, **validation_args
                        )

                with self.subTest("arbitrary heartbeat"):
                    heartbeat["updated_monotonic_ns"] = 9_000_000_000
                    heartbeat["lease_id"] = "helper-lease"
                    watchdog._write_json_atomic(
                        heartbeat_path, heartbeat
                    )
                    with self.assertRaisesRegex(
                        watchdog.LeaseValidationError,
                        "heartbeat identity",
                    ):
                        watchdog.validate_active_lease(
                            lease_path, **validation_args
                        )

                with self.subTest("outside process group"):
                    heartbeat["lease_id"] = "test-lease"
                    watchdog._write_json_atomic(
                        heartbeat_path, heartbeat
                    )
                    (process_root / str(current_pid) / "stat").write_text(
                        self._proc_stat(
                            current_pid,
                            child_pid,
                            9999,
                            456792,
                        ),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        watchdog.LeaseValidationError,
                        "outside the monitored process group",
                    ):
                        watchdog.validate_active_lease(
                            lease_path, **validation_args
                        )
                    (
                        process_root / str(current_pid) / "stat"
                    ).write_text(
                        self._proc_stat(
                            current_pid,
                            child_pid,
                            guardian_pid,
                            456792,
                        ),
                        encoding="utf-8",
                    )

                with self.subTest("environment path mismatch"):
                    bad_validation_args = {
                        **validation_args,
                        "expected_heartbeat_path": root / "other.json",
                    }
                    with self.assertRaisesRegex(
                        watchdog.LeaseValidationError,
                        "artifact paths|heartbeat path",
                    ):
                        watchdog.validate_active_lease(
                            lease_path, **bad_validation_args
                        )

                with self.subTest("lease inode mismatch"):
                    publish_lease(dict(lease))
                    lease_record = json.loads(
                        lease_path.read_text(encoding="utf-8")
                    )
                    lease_record["file_inode"] = 0
                    lease_path.write_text(
                        json.dumps(lease_record), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(
                        watchdog.LeaseValidationError,
                        "identity does not match",
                    ):
                        watchdog.validate_active_lease(
                            lease_path, **validation_args
                        )

                with self.subTest("watchdog start tick mismatch"):
                    publish_lease(dict(lease))
                    (process_root / str(watchdog_pid) / "stat").write_text(
                        self._proc_stat(
                            watchdog_pid,
                            1,
                            watchdog_pid,
                            watchdog_start_ticks + 1,
                        ),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        watchdog.LeaseValidationError,
                        "start time",
                    ):
                        watchdog.validate_active_lease(
                            lease_path, **validation_args
                        )
            finally:
                os.close(audit_descriptor)

    def test_zero_swap_gate_launches_and_propagates_child_exit(self) -> None:
        harness = Harness([snapshot(50)], FakeProcess(returncode=37))

        result = harness.run()

        self.assertEqual(result, 37)
        self.assertTrue(harness.launched)
        self.assertTrue(harness.launch_kwargs["start_new_session"])
        final = harness.records()[-1]
        self.assertEqual(final["classification"], "child_exit")
        self.assertEqual(final["total_bytes"], 200)
        self.assertEqual(final["available_bytes"], 150)
        self.assertEqual(final["used_bytes"], 50)
        self.assertEqual(final["peak_used_bytes"], 50)
        self.assertEqual(final["child_status"], "exited")
        self.assertEqual(final["process_group_status"], "leader_exited")

    def test_signaled_child_exit_uses_shell_exit_convention(self) -> None:
        harness = Harness(
            [snapshot(50)],
            FakeProcess(returncode=-signal.SIGTERM),
        )

        result = harness.run()

        self.assertEqual(result, 128 + signal.SIGTERM)

    def test_active_swap_rejects_startup_without_launch(self) -> None:
        harness = Harness(
            [snapshot(50, active_swaps=("/swapfile",))],
            FakeProcess(),
        )

        result = harness.run()

        self.assertEqual(result, watchdog.EXIT_SWAP_ACTIVE)
        self.assertFalse(harness.launched)
        self.assertEqual(
            harness.records()[-1]["classification"],
            "startup_swap_active",
        )

    def test_soft_limit_sends_sigterm(self) -> None:
        def exit_on_term(process: FakeProcess, signal_number: int) -> None:
            if signal_number == signal.SIGTERM:
                process.returncode = -signal.SIGTERM

        harness = Harness(
            [snapshot(50), snapshot(110)],
            FakeProcess(),
            signal_handler=exit_on_term,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = harness.run(
                lease_path=root / "lease.json",
                heartbeat_path=root / "heartbeat.json",
                audit_path=root / "audit.jsonl",
            )
            harness.audit.close()
            heartbeat = json.loads(
                (root / "heartbeat.json").read_text(encoding="utf-8")
            )
            lease = json.loads(
                (root / "lease.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result, watchdog.EXIT_SOFT_LIMIT)
        self.assertEqual(harness.signals, [signal.SIGTERM])
        self.assertEqual(
            harness.records()[-1]["classification"], "soft_limit"
        )
        self.assertEqual(heartbeat["state"], "final")
        self.assertEqual(heartbeat["sequence"], 3)
        self.assertEqual(lease["final"]["classification"], "soft_limit")

    def test_emergency_limit_sends_sigkill(self) -> None:
        def exit_on_kill(process: FakeProcess, signal_number: int) -> None:
            if signal_number == signal.SIGKILL:
                process.returncode = -signal.SIGKILL

        harness = Harness(
            [snapshot(50), snapshot(160)],
            FakeProcess(),
            signal_handler=exit_on_kill,
        )

        result = harness.run()

        self.assertEqual(result, watchdog.EXIT_EMERGENCY_LIMIT)
        self.assertEqual(harness.signals, [signal.SIGKILL])
        self.assertEqual(
            harness.records()[-1]["classification"], "emergency_limit"
        )

    def test_grace_timeout_escalates_to_sigkill(self) -> None:
        def ignore_term(process: FakeProcess, signal_number: int) -> None:
            if signal_number == signal.SIGKILL:
                process.returncode = -signal.SIGKILL

        harness = Harness(
            [snapshot(50), snapshot(110)],
            FakeProcess(),
            signal_handler=ignore_term,
        )

        result = harness.run()

        self.assertEqual(result, watchdog.EXIT_GRACE_TIMEOUT)
        self.assertEqual(
            harness.signals,
            [signal.SIGTERM, signal.SIGKILL],
        )
        self.assertEqual(harness.clock.value, 2.0)
        self.assertEqual(
            harness.records()[-1]["classification"], "grace_timeout"
        )

    def test_swap_appearing_during_execution_kills_group(self) -> None:
        def exit_on_kill(process: FakeProcess, signal_number: int) -> None:
            if signal_number == signal.SIGKILL:
                process.returncode = -signal.SIGKILL

        harness = Harness(
            [
                snapshot(50),
                snapshot(60, active_swaps=("/swapfile",)),
            ],
            FakeProcess(),
            signal_handler=exit_on_kill,
        )

        result = harness.run()

        self.assertEqual(result, watchdog.EXIT_SWAP_ACTIVE)
        self.assertEqual(harness.signals, [signal.SIGKILL])
        self.assertEqual(
            harness.records()[-1]["classification"], "swap_appeared"
        )

    def test_runtime_procfs_error_kills_group(self) -> None:
        def exit_on_kill(process: FakeProcess, signal_number: int) -> None:
            if signal_number == signal.SIGKILL:
                process.returncode = -signal.SIGKILL

        harness = Harness(
            [snapshot(50), watchdog.ProcfsError("missing meminfo")],
            FakeProcess(),
            signal_handler=exit_on_kill,
        )

        result = harness.run()

        self.assertEqual(result, watchdog.EXIT_PROCFS_ERROR)
        self.assertEqual(harness.signals, [signal.SIGKILL])
        self.assertEqual(
            harness.records()[-1]["classification"], "procfs_error"
        )

    def test_unexpected_monitor_error_cleans_up_process_group(self) -> None:
        def exit_on_kill(process: FakeProcess, signal_number: int) -> None:
            if signal_number == signal.SIGKILL:
                process.returncode = -signal.SIGKILL

        harness = Harness(
            [snapshot(50), RuntimeError("unexpected")],
            FakeProcess(),
            signal_handler=exit_on_kill,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = harness.run(
                lease_path=root / "lease.json",
                heartbeat_path=root / "heartbeat.json",
                audit_path=root / "audit.jsonl",
            )
            harness.audit.close()
            lease = json.loads(
                (root / "lease.json").read_text(encoding="utf-8")
            )
            persistent_records = [
                json.loads(line)
                for line in (root / "audit.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        self.assertEqual(result, watchdog.EXIT_INTERNAL_ERROR)
        self.assertEqual(
            harness.signals,
            [signal.SIGTERM, signal.SIGKILL],
        )
        final = harness.records()[-1]
        self.assertEqual(final["classification"], "internal_error")
        self.assertIn("RuntimeError: unexpected", final["error"])
        self.assertEqual(lease["final"]["classification"], "internal_error")
        self.assertEqual(
            persistent_records[-1]["classification"], "internal_error"
        )

    def test_launch_failure_is_explicit(self) -> None:
        harness = Harness([snapshot(50)], FakeProcess())

        def fail_launch(
            command: tuple[str, ...], **kwargs: Any
        ) -> FakeProcess:
            raise FileNotFoundError(2, "No such file or directory")

        setattr(harness, "launcher", fail_launch)

        result = harness.run()

        self.assertEqual(result, watchdog.EXIT_LAUNCH_ERROR)
        self.assertEqual(
            harness.records()[-1]["classification"], "launch_error"
        )


if __name__ == "__main__":
    unittest.main()

"""
TaskManager — tracks, cancels, and lists background agent tasks.

Standalone module with no Gemini dependencies. Designed so agent context
(conversation history, etc.) can be attached to tasks later via the
`context` field on each task dict.
"""

import logging
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """A tracked background task."""
    id: str
    agent: str  # agent name from agents.yaml
    command: str  # the command that ultimately ran (last attempt)
    status: str  # "running", "completed", "failed", "timed_out", "cancelled"
    output: str = ""
    thread: Optional[threading.Thread] = None
    process: Optional[subprocess.Popen] = None
    context: dict = field(default_factory=dict)  # reserved for future use
    attempts: int = 0  # how many configs were tried

    def summary(self) -> str:
        short = self.command[:80]
        if len(self.command) > 80:
            short += "..."
        return f"[{self.id}] {self.status} | {self.agent} | {short}"


class TaskManager:
    """
    Manages background agent tasks.

    Usage:
        tm = TaskManager()
        tm.set_notify(callback)  # callback(str) called when task finishes
        task = tm.start("ollama", [("cmd1", 120), ("fallback_cmd", 60)])
        print(tm.list_tasks())
        tm.cancel(task.id)
    """

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._notify: Optional[Callable[[str], None]] = None
        self._lock = threading.Lock()

    def set_notify(self, callback: Callable[[str], None]):
        """Set the notification callback. Called from any thread when a task finishes."""
        self._notify = callback

    def start(self, agent: str, run_list: list[tuple[str, int, str]],
              context: dict | None = None) -> Task:
        """
        Start a background task with fallback support.

        run_list is an ordered list of (command, timeout, label) tuples.
        label describes the config (e.g. "ollama/qwen3.5:9b-64K").
        Each is tried in sequence until one succeeds (exit code 0 + non-empty output).
        If all fail, the task is marked failed with all errors.
        """
        task_id = uuid.uuid4().hex[:8]
        task = Task(
            id=task_id,
            agent=agent,
            command=run_list[0][0] if run_list else "",
            status="running",
            context=context or {},
            attempts=len(run_list),
        )

        with self._lock:
            self._tasks[task_id] = task

        thread = threading.Thread(
            target=self._run_task_with_fallbacks,
            args=(task, run_list),
            daemon=True,
        )
        task.thread = thread
        thread.start()
        logger.info(f"Started task {task_id}: agent={agent} attempts={len(run_list)}")
        return task

    def _run_task_with_fallbacks(self, task: Task, run_list: list[tuple[str, int, str]]):
        """Try each (command, timeout, label) in order until one succeeds."""
        failures = []  # list of (label, error)

        for i, (command, timeout, label) in enumerate(run_list):
            is_primary = (i == 0)

            with self._lock:
                # Don't try more if cancelled
                if task.status != "running":
                    return
                task.command = command if isinstance(command, str) else f"<callable:{label}>"

            cmd_preview = command[:100] if isinstance(command, str) else f"<callable:{label}>"
            logger.info(f"Task {task.id} [{label}] attempt {'primary' if is_primary else f'fallback {i}'}: {cmd_preview}")

            success, output = self._run_single(command, timeout)

            if success:
                with self._lock:
                    if task.status != "running":
                        return
                    task.output = output
                    task.status = "completed"
                    if failures:
                        failed_summary = ", ".join(f"{lbl} ({err[:50]})" for lbl, err in failures)
                        task.output = f"(succeeded on {label} after: {failed_summary})\n{output}"
                logger.info(f"Task {task.id} completed on {label}: {output[:200]}")
                self._send_notification(task)
                return

            # Failed — record and try next
            logger.warning(f"Task {task.id} [{label}] failed: {output[:200]}")
            failures.append((label, output))

        # All attempts exhausted
        with self._lock:
            if task.status != "running":
                return
            failure_details = "\n".join(f"  [{lbl}] {err[:200]}" for lbl, err in failures)
            task.output = f"All {len(run_list)} attempt(s) failed:\n{failure_details}"
            task.status = "failed"

        logger.error(f"Task {task.id} all {len(run_list)} attempts failed")
        self._send_notification(task)

    def _run_single(self, command, timeout):
        """
        Run a command or callable. Returns (success, output).
        success = True if exit code 0 and non-empty stdout (for shell)
                  or non-empty result (for callable).
        """
        if callable(command):
            return self._run_callable(command)

        try:
            proc = subprocess.Popen(
                command, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            stdout, stderr = proc.communicate(timeout=timeout)

            output = stdout or ""
            if stderr:
                # Filter out known noisy warnings from agent CLIs
                noisy = [
                    "Warning: no stdin data received",
                    "proceeding without it",
                    "redirect stdin explicitly",
                ]
                filtered = "\n".join(
                    line for line in stderr.strip().splitlines()
                    if not any(n in line for n in noisy)
                )
                if filtered.strip():
                    output += "\nSTDERR: " + filtered
            if proc.returncode != 0:
                output += f"\nExit code: {proc.returncode}"

            output = output.strip() or "(no output)"
            success = proc.returncode == 0 and bool(stdout and stdout.strip())
            return success, output

        except subprocess.TimeoutExpired:
            proc.kill()
            return False, f"Timed out after {timeout}s"
        except Exception as e:
            return False, str(e)

    def _run_callable(self, func):
        """Run a Python callable (e.g. Ollama tool loop) and return (success, output)."""
        try:
            output = func()
            output = str(output or "").strip()
            success = bool(output) and not output.startswith("Error:")
            return success, output or "(no output)"
        except Exception as e:
            return False, str(e)

    def _send_notification(self, task: Task):
        """Fire the notification callback if set."""
        if self._notify:
            msg = (
                f"[Task {task.id} {task.status}] "
                f"Agent: {task.agent} | Attempts: {task.attempts}\n"
                f"Output:\n{task.output[:1000]}"
            )
            try:
                self._notify(msg)
            except Exception as e:
                logger.error(f"Notification callback error: {e}")

    def cancel(self, task_id: str) -> str:
        """Cancel a running task by ID."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return f"Task {task_id} not found"
            if task.status != "running":
                return f"Task {task_id} is {task.status}, nothing to cancel"

            if task.process:
                task.process.kill()
            task.status = "cancelled"
            task.output = "Cancelled by user"

        logger.info(f"Cancelled task {task_id}")
        return f"Task {task_id} cancelled"

    def list_tasks(self, status_filter: str | None = None) -> list[dict]:
        """Return task summaries. Optionally filter by status."""
        with self._lock:
            tasks = list(self._tasks.values())
        if status_filter:
            tasks = [t for t in tasks if t.status == status_filter]
        return [
            {"id": t.id, "agent": t.agent, "status": t.status,
             "command": t.command, "output": t.output[:500]}
            for t in tasks
        ]

    def get_task(self, task_id: str) -> Optional[dict]:
        """Get a single task's details."""
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return None
        return {
            "id": task.id, "agent": task.agent, "status": task.status,
            "command": task.command, "output": task.output,
            "context": task.context, "attempts": task.attempts,
        }

    def cleanup(self, max_age: int = 100):
        """Remove old completed/failed/cancelled tasks to prevent unbounded growth."""
        with self._lock:
            done = [t for t in self._tasks.values() if t.status != "running"]
            if len(done) > max_age:
                keep_ids = {t.id for t in done[-max_age:]}
                running_ids = {t.id for t in self._tasks.values() if t.status == "running"}
                to_remove = [tid for tid in self._tasks if tid not in keep_ids and tid not in running_ids]
                for tid in to_remove:
                    del self._tasks[tid]
                logger.info(f"Cleaned up {len(to_remove)} old tasks")

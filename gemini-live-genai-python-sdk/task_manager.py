"""
TaskManager — tracks, cancels, and lists background agent tasks.

Standalone module with no Gemini dependencies. Designed so agent context
(conversation history, etc.) can be attached to tasks later via the
`context` field on each task dict.
"""

import asyncio
import logging
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """A tracked background task."""
    id: str
    agent: str  # agent name from agents.yaml
    command: str
    status: str  # "running", "completed", "failed", "timed_out", "cancelled"
    output: str = ""
    thread: Optional[threading.Thread] = None
    process: Optional[subprocess.Popen] = None
    context: dict = field(default_factory=dict)  # reserved for future use

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
        task = tm.start("ollama", "scripts/ask_ollama 'hello'")
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

    def start(self, agent: str, command: str, timeout: int = 120,
              context: dict | None = None) -> Task:
        """
        Start a background task. Returns immediately with a Task object.
        """
        task_id = uuid.uuid4().hex[:8]
        task = Task(
            id=task_id,
            agent=agent,
            command=command,
            status="running",
            context=context or {},
        )

        with self._lock:
            self._tasks[task_id] = task

        thread = threading.Thread(
            target=self._run_task,
            args=(task, command, timeout),
            daemon=True,
        )
        task.thread = thread
        thread.start()
        logger.info(f"Started task {task_id}: agent={agent} cmd={command[:80]}")
        return task

    def _run_task(self, task: Task, command: str, timeout: int):
        """Internal: run a command in a background thread."""
        try:
            proc = subprocess.Popen(
                command, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            with self._lock:
                task.process = proc

            stdout, stderr = proc.communicate(timeout=timeout)

            with self._lock:
                # Don't overwrite if already cancelled
                if task.status != "running":
                    return

                output = stdout or ""
                if stderr:
                    output += "\nSTDERR: " + stderr
                if proc.returncode != 0:
                    output += f"\nExit code: {proc.returncode}"
                task.output = output.strip() or "(no output)"
                task.status = "failed" if proc.returncode != 0 else "completed"

            logger.info(f"Task {task.id} {task.status}: {output[:200]}")
            self._send_notification(task)

        except subprocess.TimeoutExpired:
            with self._lock:
                if task.status == "running":
                    proc.kill()
                    task.output = f"Timed out after {timeout}s"
                    task.status = "timed_out"
            logger.warning(f"Task {task.id} timed out after {timeout}s")
            self._send_notification(task)

        except Exception as e:
            with self._lock:
                if task.status == "running":
                    task.output = str(e)
                    task.status = "failed"
            logger.error(f"Task {task.id} error: {e}")
            self._send_notification(task)

    def _send_notification(self, task: Task):
        """Fire the notification callback if set."""
        if self._notify:
            msg = (
                f"[Task {task.id} {task.status}] "
                f"Agent: {task.agent} | Command: {task.command[:100]}\n"
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
            "context": task.context,
        }

    def cleanup(self, max_age: int = 100):
        """Remove old completed/failed/cancelled tasks to prevent unbounded growth."""
        with self._lock:
            done = [t for t in self._tasks.values() if t.status != "running"]
            # Keep the most recent max_age finished tasks
            if len(done) > max_age:
                keep_ids = {t.id for t in done[-max_age:]}
                running_ids = {t.id for t in self._tasks.values() if t.status == "running"}
                to_remove = [tid for tid in self._tasks if tid not in keep_ids and tid not in running_ids]
                for tid in to_remove:
                    del self._tasks[tid]
                logger.info(f"Cleaned up {len(to_remove)} old tasks")

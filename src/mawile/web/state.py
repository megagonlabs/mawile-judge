from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import yaml

from mawile.config_payload import load_config_payload
from mawile.pipeline import AuditPipeline, AuditRunResult
from mawile.project import PROJECT_ROOT, discover_run_artifacts
from mawile.run_artifacts import (
    ARTIFACT_FILES,
    RunBundle,
    discover_run_dir,
    is_relative_to,
    load_run_bundle,
)
from mawile.schemas import AuditRunConfig


@dataclass
class AuditJob:
    job_id: str
    status: str = "queued"
    progress: Queue[dict[str, Any]] = field(default_factory=Queue)
    result: AuditRunResult | None = None
    error: str | None = None

    def push(self, event: dict[str, Any]) -> None:
        self.progress.put(event)


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.config_payload: dict[str, Any] = {}
        self.config_base_dir = PROJECT_ROOT
        self.config_source_path: Path | None = None
        self.config_uploaded = False
        self.last_planned_operators: dict[str, Any] | None = None
        self.current_run_id: str | None = None
        self.jobs: dict[str, AuditJob] = {}
        self.run_dirs: dict[str, Path] = {}

    def load_config_path(self, path: Path) -> None:
        loaded = load_config_payload(path)
        with self.lock:
            self.config_payload = loaded.payload
            self.config_base_dir = loaded.base_dir
            self.config_source_path = loaded.source_path
            self.config_uploaded = False
            self.last_planned_operators = None

    def load_uploaded_yaml(self, raw_text: str) -> None:
        payload = yaml.safe_load(raw_text) or {}
        if not isinstance(payload, dict):
            raise ValueError("Uploaded YAML must contain a mapping.")
        with self.lock:
            self.config_payload = payload
            self.config_base_dir = PROJECT_ROOT
            self.config_source_path = None
            self.config_uploaded = True
            self.last_planned_operators = None

    def update_payload(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.config_payload = payload

    def update_planned(self, payload: dict[str, Any], selection: Any) -> None:
        with self.lock:
            self.config_payload = payload
            self.last_planned_operators = {
                "selected": list(selection.selected),
                "manual": list(selection.manual),
                "effective": list(selection.effective),
                "selected_directional": list(selection.selected_directional),
                "manual_directional": list(selection.manual_directional),
                "effective_directional": list(selection.effective_directional),
                "rationale": selection.rationale,
            }

    def start_audit(self, config: AuditRunConfig) -> AuditJob:
        job = AuditJob(job_id=uuid.uuid4().hex)
        with self.lock:
            self.jobs[job.job_id] = job
        thread = threading.Thread(target=self._run_job, args=(job, config), daemon=True)
        thread.start()
        return job

    def _run_job(self, job: AuditJob, config: AuditRunConfig) -> None:
        job.status = "running"

        def progress(event: dict[str, Any]) -> None:
            job.push(event)

        try:
            result = AuditPipeline(config, progress_callback=progress).run()
        except Exception as exc:  # noqa: BLE001 - surfaced in the local UI.
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.push({"stage": "error", "message": job.error})
            return

        job.result = result
        job.status = "complete"
        self.remember_run(result.run_id, result.paths["run_dir"])
        job.push({"stage": "complete", "message": "Audit complete", "run_id": result.run_id})

    def remember_run(self, run_id: str, run_dir: Path) -> None:
        with self.lock:
            self.current_run_id = run_id
            self.run_dirs[run_id] = run_dir.resolve()

    def load_run(self, run_id: str | None = None) -> RunBundle | None:
        selected_id = run_id or self.current_run_id
        if not selected_id:
            latest = next(iter(discover_run_artifacts()), None)
            selected_id = latest.parent.name if latest else None
        if not selected_id:
            return None

        run_dir = self.run_dirs.get(selected_id) or discover_run_dir(selected_id)
        if run_dir is None:
            return None
        bundle = load_run_bundle(run_dir)
        self.remember_run(bundle.run_id, bundle.run_dir)
        return bundle

    def artifact_path(self, run_id: str, artifact_key: str) -> tuple[Path, str] | None:
        entry = ARTIFACT_FILES.get(artifact_key)
        if entry is None:
            return None
        bundle = self.load_run(run_id)
        if bundle is None:
            return None
        rel_path, media_type = entry
        path = (bundle.run_dir / rel_path).resolve()
        if not is_relative_to(path, bundle.run_dir.resolve()) or not path.exists():
            return None
        return path, media_type


def drain_job_events(job: AuditJob) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    while True:
        try:
            events.append(job.progress.get_nowait())
        except Empty:
            return events


def event_json(event: dict[str, Any]) -> str:
    return json.dumps(event, sort_keys=True)

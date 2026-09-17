from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from mawile.env import load_environment
from mawile.config_payload import build_config_from_payload
from mawile.project import discover_config_files
from mawile.suggestions import (
    plan_perturbation_operators,
    suggest_custom_perturbation,
)
from mawile.web.forms import FormErrors, payload_from_form
from mawile.web.state import AppState, drain_job_events, event_json
from mawile.web.views import (
    catalog_context,
    configure_context,
    family_accent,
    family_label,
    fmt_float,
    fmt_signed,
    fmt_usd,
    report_context,
    result_label,
    result_score,
    selected_judge_results,
    title_label,
    validation_label,
)

PACKAGE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))
templates.env.filters["family_label"] = family_label
templates.env.filters["family_accent"] = family_accent
templates.env.filters["title_label"] = title_label
templates.env.filters["fmt_float"] = fmt_float
templates.env.filters["fmt_signed"] = fmt_signed
templates.env.filters["fmt_usd"] = fmt_usd
templates.env.filters["validation_label"] = validation_label
templates.env.globals["selected_judge_results"] = selected_judge_results
templates.env.globals["result_label"] = result_label
templates.env.globals["result_score"] = result_score


def create_app(state: AppState | None = None) -> FastAPI:
    load_environment()
    app = FastAPI(title="MAWILE UI")
    app.state.mawile_state = state or AppState()
    app.mount(
        "/static",
        StaticFiles(directory=str(PACKAGE_DIR / "static")),
        name="static",
    )

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/configure", status_code=303)

    @app.get("/configure", response_class=HTMLResponse)
    async def configure(request: Request, job: str | None = None) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "configure.html",
            configure_context(_state(request), job_id=job),
        )

    @app.get("/catalog", response_class=HTMLResponse)
    async def catalog(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "catalog.html",
            catalog_context(_state(request)),
        )

    @app.post("/configure", response_class=HTMLResponse)
    async def update_config(request: Request) -> Any:
        form = await request.form()
        state = _state(request)
        try:
            payload = payload_from_form(state.config_payload, form)
        except FormErrors as exc:
            return templates.TemplateResponse(
                request,
                "configure.html",
                configure_context(state, errors=exc.errors),
                status_code=400,
            )
        state.update_payload(payload)
        return RedirectResponse("/configure", status_code=303)

    @app.post("/configure/autosave")
    async def autosave_config(request: Request) -> JSONResponse:
        form = await request.form()
        state = _state(request)
        try:
            payload = payload_from_form(state.config_payload, form)
        except FormErrors as exc:
            return JSONResponse({"ok": False, "errors": exc.errors}, status_code=422)
        state.update_payload(payload)
        return JSONResponse({"ok": True})

    @app.post("/config/load")
    async def load_config(request: Request) -> RedirectResponse:
        form = await request.form()
        selected = Path(str(form.get("config_path", ""))).expanduser().resolve()
        known = {path.resolve() for path in discover_config_files()}
        if selected not in known:
            raise HTTPException(status_code=400, detail="Unknown config path.")
        _state(request).load_config_path(selected)
        return RedirectResponse("/configure", status_code=303)

    @app.post("/config/upload")
    async def upload_config(request: Request, config_file: UploadFile = File(...)) -> Any:
        raw = (await config_file.read()).decode("utf-8")
        state = _state(request)
        try:
            state.load_uploaded_yaml(raw)
        except (yaml.YAMLError, ValueError) as exc:
            return templates.TemplateResponse(
                request,
                "configure.html",
                configure_context(state, errors=[f"Config upload failed: {exc}"]),
                status_code=400,
            )
        return RedirectResponse("/configure", status_code=303)

    @app.post("/planner/suggest", response_class=HTMLResponse)
    async def suggest_operators(request: Request) -> Any:
        form = await request.form()
        state = _state(request)
        try:
            payload = payload_from_form(state.config_payload, form)
            planned_payload, selection = plan_perturbation_operators(
                payload,
                state.config_base_dir,
            )
        except FormErrors as exc:
            return templates.TemplateResponse(
                request,
                "configure.html",
                configure_context(state, errors=exc.errors),
                status_code=400,
            )
        except Exception as exc:  # noqa: BLE001 - suggestion failure belongs in the UI.
            state.update_payload(payload if "payload" in locals() else state.config_payload)
            return templates.TemplateResponse(
                request,
                "configure.html",
                configure_context(
                    state,
                    errors=[f"Perturbation Suggester could not generate suggestions: {exc}"],
                ),
                status_code=400,
            )
        state.update_planned(planned_payload, selection)
        return RedirectResponse("/configure", status_code=303)

    @app.post("/custom-perturbations/suggest", response_class=HTMLResponse)
    async def suggest_custom(request: Request) -> Any:
        form = await request.form()
        state = _state(request)
        form_payload = _mutable_form_mapping(form)
        row_index = _custom_suggest_index(form_payload)
        try:
            payload = payload_from_form(state.config_payload, form_payload)
            target = str(form_payload.get(f"custom_{row_index}_target") or "item.input")
            expected_effect = str(
                form_payload.get(f"custom_{row_index}_expected_effect") or "same_verdict"
            )
            suggestion = suggest_custom_perturbation(
                payload,
                state.config_base_dir,
                target=target,
                expected_effect=expected_effect,
            )
            form_payload[f"custom_{row_index}_name"] = suggestion.name
            form_payload[f"custom_{row_index}_target"] = suggestion.target
            form_payload[f"custom_{row_index}_expected_effect"] = suggestion.expected_effect
            form_payload[f"custom_{row_index}_instruction"] = suggestion.instruction
            form_payload[f"custom_{row_index}_enabled"] = "on"
            payload = payload_from_form(state.config_payload, form_payload)
        except FormErrors as exc:
            return templates.TemplateResponse(
                request,
                "configure.html",
                configure_context(state, errors=exc.errors),
                status_code=400,
            )
        except Exception as exc:  # noqa: BLE001 - suggestion failure belongs in the UI.
            state.update_payload(payload if "payload" in locals() else state.config_payload)
            return templates.TemplateResponse(
                request,
                "configure.html",
                configure_context(
                    state,
                    errors=[f"Could not suggest a custom perturbation: {exc}"],
                ),
                status_code=400,
            )
        state.update_payload(payload)
        return RedirectResponse("/configure", status_code=303)

    @app.post("/runs/start", response_class=HTMLResponse)
    async def start_run(request: Request) -> Any:
        form = await request.form()
        state = _state(request)
        try:
            payload = payload_from_form(state.config_payload, form)
            config = build_config_from_payload(payload, state.config_base_dir)
        except FormErrors as exc:
            return templates.TemplateResponse(
                request,
                "configure.html",
                configure_context(state, errors=exc.errors),
                status_code=400,
            )
        except Exception as exc:  # noqa: BLE001 - validation is shown inline.
            return templates.TemplateResponse(
                request,
                "configure.html",
                configure_context(state, errors=[f"Config is not runnable: {exc}"]),
                status_code=400,
            )
        state.update_payload(payload)
        job = state.start_audit(config)
        return RedirectResponse(f"/configure?job={job.job_id}", status_code=303)

    @app.get("/runs/events/{job_id}")
    async def run_events(request: Request, job_id: str) -> StreamingResponse:
        job = _state(request).jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown run job.")

        async def stream():
            while True:
                for event in drain_job_events(job):
                    yield f"data: {event_json(event)}\n\n"
                if job.status in {"complete", "error"}:
                    break
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/report", response_class=HTMLResponse)
    async def report(
        request: Request,
        run: str | None = None,
        operator: str | None = None,
        item: str | None = None,
        directional_operator: str | None = None,
    ) -> Any:
        state = _state(request)
        bundle = state.load_run(run)
        if bundle is None:
            return RedirectResponse("/configure", status_code=303)
        return templates.TemplateResponse(
            request,
            "report.html",
            report_context(
                state,
                bundle,
                selected_operator=operator,
                selected_item=item,
                selected_directional_operator=directional_operator,
            ),
        )

    @app.get("/artifacts/{run_id}/{artifact_key}")
    async def artifact(request: Request, run_id: str, artifact_key: str) -> FileResponse:
        resolved = _state(request).artifact_path(run_id, artifact_key)
        if resolved is None:
            raise HTTPException(status_code=404, detail="Artifact not found.")
        path, media_type = resolved
        return FileResponse(path, media_type=media_type, filename=path.name)

    return app


def _state(request: Request) -> AppState:
    return request.app.state.mawile_state


def _mutable_form_mapping(form: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if hasattr(form, "multi_items"):
        items = form.multi_items()
    elif isinstance(form, dict):
        items = form.items()
    else:
        items = []
    for key, value in items:
        if key in values:
            current = values[key]
            if isinstance(current, list):
                current.append(value)
            else:
                values[key] = [current, value]
        else:
            values[key] = value
    return values


def _custom_suggest_index(form: dict[str, Any]) -> int:
    try:
        index = int(str(form.get("custom_suggest_index", "0")))
    except (TypeError, ValueError):
        index = 0
    try:
        count = int(str(form.get("custom_count", "0") or "0"))
    except (TypeError, ValueError):
        count = 0
    return max(0, min(index, max(count - 1, 0)))


app = create_app()

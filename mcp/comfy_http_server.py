from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from mcp.server.mcpserver import MCPServer


COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
HTTP_TIMEOUT_SECONDS = 30.0
MAX_WAIT_SECONDS = 3600.0
DOWNLOAD_TIMEOUT_SECONDS = 1800.0
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
MODELS_ROOT = Path(os.environ.get("COMFYUI_MODELS_DIR", "/workspace/ComfyUI/models"))
TEMPLATES_ROOT = Path(
    os.environ.get("COMFYUI_TEMPLATES_DIR", "/workspace/ComfyUI/user/default/workflows")
)
TEMPLATE_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
MODEL_FILE_SUFFIXES = (
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".gguf",
    ".sft",
)

# Keyword -> models/ sub-directory, mirroring StoryOS's Comfy_Manifest::MODEL_FIELDS
# so a downloaded file lands where the matching loader node expects it.
MODEL_FOLDER_HINTS = (
    ("vae", "vae"),
    ("lora", "loras"),
    ("controlnet", "controlnet"),
    ("control_net", "controlnet"),
    ("clip_vision", "clip_vision"),
    ("clip", "text_encoders"),
    ("gligen", "gligen"),
    ("upscale", "upscale_models"),
    ("style_model", "style_models"),
    ("unet", "diffusion_models"),
    ("diffusion_model", "diffusion_models"),
)
DEFAULT_MODEL_FOLDER = "checkpoints"

mcp = MCPServer(
    "comfyui-http",
    instructions=(
        "Discover workflows with list_templates and get_template, submit "
        "API-format ComfyUI workflows with submit_workflow, then use "
        "wait_for_result or get_history with the returned prompt_id. Use "
        "download_models to fetch missing checkpoint/model files into ComfyUI's "
        "models directory."
    ),
)


def _request_json(
    method: str, path: str, payload: dict[str, Any] | None = None
) -> Any:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"{COMFYUI_URL}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )

    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return json.load(response)
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        try:
            detail: Any = json.loads(response_body)
        except json.JSONDecodeError:
            detail = response_body
        raise RuntimeError(
            f"ComfyUI {method} {path} failed with HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach ComfyUI at {COMFYUI_URL}: {exc.reason}") from exc


def _history(prompt_id: str) -> dict[str, Any] | None:
    history = _request_json("GET", f"/history/{quote(prompt_id, safe='')}")
    if not isinstance(history, dict):
        raise RuntimeError("ComfyUI returned an invalid history response")
    entry = history.get(prompt_id)
    return entry if isinstance(entry, dict) else None


@mcp.tool()
async def submit_workflow(
    workflow: dict[str, Any],
    client_id: str | None = None,
) -> dict[str, Any]:
    """Submit an API-format workflow graph to ComfyUI's POST /prompt endpoint."""
    payload: dict[str, Any] = {"prompt": workflow}
    if client_id:
        payload["client_id"] = client_id

    result = await asyncio.to_thread(_request_json, "POST", "/prompt", payload)
    if not isinstance(result, dict) or not isinstance(result.get("prompt_id"), str):
        raise RuntimeError(f"ComfyUI returned an invalid prompt response: {result!r}")
    return result


@mcp.tool()
async def get_history(prompt_id: str) -> dict[str, Any]:
    """Read one prompt's current /history result without waiting."""
    entry = await asyncio.to_thread(_history, prompt_id)
    return {"prompt_id": prompt_id, "found": entry is not None, "result": entry}


def _workflow_format(workflow: Any) -> str:
    if isinstance(workflow, dict):
        if isinstance(workflow.get("nodes"), list):
            return "ui"
        if any(isinstance(node, dict) and "class_type" in node for node in workflow.values()):
            return "api"
    return "unknown"


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _node_class_types(workflow: Any, workflow_format: str) -> list[str]:
    classes: set[str] = set()
    if workflow_format == "api" and isinstance(workflow, dict):
        for node in workflow.values():
            if isinstance(node, dict) and isinstance(node.get("class_type"), str):
                classes.add(node["class_type"])
    elif workflow_format == "ui" and isinstance(workflow, dict):
        for node in workflow.get("nodes", []):
            if isinstance(node, dict) and isinstance(node.get("type"), str):
                classes.add(node["type"])
    return sorted(classes)


def _installed_model_path(filename: str) -> str | None:
    if not MODELS_ROOT.is_dir():
        return None
    for candidate in MODELS_ROOT.rglob(filename):
        if candidate.is_file():
            return str(candidate)
    return None


def _referenced_models(workflow: Any) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for text in _walk_strings(workflow):
        if not text.lower().endswith(MODEL_FILE_SUFFIXES):
            continue
        reference = text.replace("\\", "/")
        filename = os.path.basename(reference)
        if not filename or filename in seen:
            continue
        installed = _installed_model_path(filename)
        seen[filename] = {
            "reference": reference,
            "filename": filename,
            "expected_folder": _folder_for_filename(filename),
            "installed": installed is not None,
            "path": installed,
        }
    return [seen[key] for key in sorted(seen)]


def _template_files() -> list[Path]:
    if not TEMPLATES_ROOT.is_dir():
        return []
    return sorted(path for path in TEMPLATES_ROOT.glob("*.json") if path.is_file())


def _read_template(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _list_templates() -> dict[str, Any]:
    templates: list[dict[str, Any]] = []
    for path in _template_files():
        entry: dict[str, Any] = {
            "id": path.stem,
            "name": path.stem.replace("-", " ").replace("_", " "),
            "path": str(path),
            "bytes": path.stat().st_size,
            "modified": int(path.stat().st_mtime),
        }
        try:
            workflow = _read_template(path)
        except (OSError, json.JSONDecodeError) as exc:
            entry["format"] = "invalid"
            entry["error"] = str(exc)
        else:
            entry["format"] = _workflow_format(workflow)
            entry["node_count"] = (
                len(workflow.get("nodes", []))
                if entry["format"] == "ui"
                else len(workflow)
                if isinstance(workflow, dict)
                else 0
            )
        templates.append(entry)
    return {"root": str(TEMPLATES_ROOT), "templates": templates}


def _get_template(template_id: str) -> dict[str, Any]:
    if not TEMPLATE_ID_PATTERN.fullmatch(template_id) or template_id in (".", ".."):
        raise ValueError(f"Invalid template id: {template_id!r}")

    path = TEMPLATES_ROOT / f"{template_id}.json"
    if not path.is_file():
        raise RuntimeError(f"Template {template_id!r} was not found in {TEMPLATES_ROOT}")

    workflow = _read_template(path)
    workflow_format = _workflow_format(workflow)
    models = _referenced_models(workflow)
    return {
        "id": template_id,
        "name": template_id.replace("-", " ").replace("_", " "),
        "path": str(path),
        "format": workflow_format,
        "workflow": workflow,
        "requirements": {
            "node_classes": _node_class_types(workflow, workflow_format),
            "models": models,
            "missing_models": [m["filename"] for m in models if not m["installed"]],
        },
    }


@mcp.tool()
async def list_templates() -> dict[str, Any]:
    """List the workflow templates available to this ComfyUI installation."""
    return await asyncio.to_thread(_list_templates)


@mcp.tool()
async def get_template(template_id: str) -> dict[str, Any]:
    """Return one template's workflow graph plus its node and model requirements."""
    return await asyncio.to_thread(_get_template, template_id)


def _folder_for_filename(filename: str) -> str:
    lowered = filename.lower()
    for keyword, folder in MODEL_FOLDER_HINTS:
        if keyword in lowered:
            return folder
    return DEFAULT_MODEL_FOLDER


def _filename_from_url(url: str) -> str:
    name = os.path.basename(urlparse(url).path)
    if not name:
        raise RuntimeError(f"Cannot determine a filename from URL: {url}")
    return name


def _download_one(url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise RuntimeError(f"Refusing non-http(s) URL: {url}")

    filename = _filename_from_url(url)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", filename):
        raise RuntimeError(f"Refusing unsafe filename derived from URL: {filename!r}")

    folder = _folder_for_filename(filename)
    destination_dir = MODELS_ROOT / folder
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / filename

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
    headers = {"User-Agent": "storyos-comfyui-mcp/1.0"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    request = Request(url, headers=headers)
    with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        resumed = resume_from > 0 and response.status == 206
        expected_size = response.headers.get("Content-Length")
        expected_size = int(expected_size) if expected_size and expected_size.isdigit() else None
        if resumed:
            content_range = response.headers.get("Content-Range", "")
            match = re.search(r"/([0-9]+)$", content_range)
            expected_size = int(match.group(1)) if match else (
                expected_size + resume_from if expected_size is not None else None
            )

        if expected_size is not None and destination.exists() and destination.stat().st_size == expected_size:
            return {
                "url": url,
                "filename": filename,
                "folder": folder,
                "path": str(destination),
                "status": "already_installed",
                "bytes": expected_size,
            }

        written = resume_from if resumed else 0
        with open(tmp_path, "ab" if resumed else "wb") as handle:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
        tmp_path.replace(destination)

    return {
        "url": url,
        "filename": filename,
        "folder": folder,
        "path": str(destination),
        "status": "downloaded",
        "bytes": written,
    }


@mcp.tool()
async def download_models(urls: list[str]) -> dict[str, Any]:
    """Download model files by URL into the matching models/<folder> ComfyUI's
    loader nodes expect, inferring the folder from each filename. Skips a file
    that already exists locally with a matching size."""
    if not urls:
        raise ValueError("urls must contain at least one download URL")

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for url in urls:
        try:
            results.append(await asyncio.to_thread(_download_one, url))
        except Exception as exc:  # noqa: BLE001 - report per-file failures, keep going
            errors.append({"url": url, "error": str(exc)})

    return {"installed": results, "errors": errors, "ok": not errors}


@mcp.tool()
async def wait_for_result(
    prompt_id: str,
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 2.0,
) -> dict[str, Any]:
    """Poll /history until a prompt completes or fails, returning its full result."""
    if not 0.0 < timeout_seconds <= MAX_WAIT_SECONDS:
        raise ValueError(f"timeout_seconds must be between 0 and {MAX_WAIT_SECONDS}")
    if not 0.1 <= poll_interval_seconds <= 30.0:
        raise ValueError("poll_interval_seconds must be between 0.1 and 30")

    deadline = time.monotonic() + timeout_seconds
    while True:
        entry = await asyncio.to_thread(_history, prompt_id)
        if entry is not None:
            return {"prompt_id": prompt_id, "result": entry}

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Prompt {prompt_id!r} did not appear in history within "
                f"{timeout_seconds} seconds"
            )
        await asyncio.sleep(min(poll_interval_seconds, remaining))


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "streamable-http":
        mcp.run(
            transport="streamable-http",
            host=os.environ.get("MCP_HTTP_HOST", "0.0.0.0"),
            port=int(os.environ.get("MCP_HTTP_PORT", "9000")),
        )
    else:
        mcp.run(transport="stdio")
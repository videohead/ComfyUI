# ComfyUI Agent Instructions

This directory is the local Docker deployment of ComfyUI — the node-based
generative image and video server — plus its MCP adapter. The generation
server runs as container `comfyui` (HTTP API on `127.0.0.1:8188`, internal
`comfyui:8188`); the MCP adapter runs as `comfyui-mcp`
(`127.0.0.1:9000`, internal `comfyui-mcp:9000`) and is registered as a
metis-router upstream (`comfyui:` tool namespace).

World Graph Studio and other clients treat ComfyUI as two services: the HTTP
API executes workflows and returns media; the MCP service provides template
discovery and model download operations. The port-8188 HTTP server does not
speak MCP — never append `/mcp` to it.

## Tool execution rule

All tool calls that invoke `python`, `node`, `vite`, or `php` MUST run inside
Docker — never on the host. The host has no project runtimes installed.

- Python runs in the `comfyui` container, e.g.
  `docker compose -f /opt/ComfyUI/docker-compose.yml exec comfyui python ...`
- For ad-hoc Node tooling, use
  `docker run --rm -v /opt/ComfyUI:/srv -w /srv node:22-alpine ...`

## Layout

- `Dockerfile`, `docker-compose.yml`, `entrypoint.sh` — the deployment
- `custom_nodes/` — installed custom node packs
- `models/` — checkpoints and model weights
- `input/`, `output/`, `temp_downloads/` — generation I/O
- `mcp/` — the MCP adapter sources
- `ltx-text-to-video.json` — bundled workflow example
- `MCP_Configuration.md` — MCP connection model details

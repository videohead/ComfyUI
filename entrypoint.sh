#!/usr/bin/env bash
set -euo pipefail

# Container's main process is ComfyUI itself; comfy-mcp is also available over
# stdio via `docker exec` for local MCP clients (see docker-compose.yml comments).
# Any explicit command (e.g. `docker run ... comfy install --help`) bypasses the default launch.
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

# Streamable-HTTP MCP is a separate compose service (comfyui-mcp) that runs this
# same image with `python3 /opt/comfy-http-mcp/server.py`.
exec comfy --workspace "${COMFY_WORKSPACE}" launch -- \
    --listen 0.0.0.0 \
    --port 8188

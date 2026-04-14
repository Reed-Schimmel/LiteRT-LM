# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP server for LiteRT-LM with Gemini-compatible API.

Reference: https://ai.google.dev/api/generate-content
"""

import http.server
import json
import re
from typing import Any, Optional

import click

import litert_lm
from litert_lm_cli import model

GEN_CONTENT_RE = re.compile(r"^/v1beta/models/([^/\\:]+):generateContent$")
STREAM_GEN_CONTENT_RE = re.compile(r"^/v1beta/models/([^/\\:]+):streamGenerateContent$")

_current_engine: Optional[litert_lm.Engine] = None
_current_model_id: Optional[str] = None


def get_engine(model_id: str, backend_str: str = "cpu") -> litert_lm.Engine:
    """Gets or creates the LiteRT-LM engine for the given model ID.

    The LiteRT-LM Engine is a globally cached resource. Its lifetime is managed
    manually:
    -   `__enter__` is called when a new engine is created for a `model_id`.
    -   `__exit__` is called when switching to a different `model_id` or when the
        server is shutting down.

    Args:
      model_id: The identifier for the model.
      backend_str: The backend to use (e.g. 'cpu' or 'gpu').

    Returns:
      The initialized LiteRT-LM Engine.

    Raises:
      FileNotFoundError: If the model for the given `model_id` is not found.
    """
    global _current_engine, _current_model_id
    if _current_model_id == model_id and _current_engine is not None:
        return _current_engine

    # If we are switching models or re-initializing, clear the old one first.
    if _current_engine is not None:
        _current_engine.__exit__(None, None, None)
        _current_engine = None
        _current_model_id = None

    m = model.Model.from_model_id(model_id)
    if not m.exists():
        raise FileNotFoundError(f"Model {model_id} not found")

    click.echo(
        click.style(
            f"Initializing engine for model: {m.model_path} with backend: {backend_str}",
            fg="cyan",
        )
    )
    backend = (
        litert_lm.Backend.GPU if backend_str.lower() == "gpu" else litert_lm.Backend.CPU
    )
    new_engine = litert_lm.Engine(m.model_path, backend=backend)
    new_engine.__enter__()

    _current_engine = new_engine
    _current_model_id = model_id
    return _current_engine


def gemini_to_litertlm_message(
    gemini_content: dict[str, Any],
) -> dict[str, Any]:
    """Converts a Gemini API content object to a LiteRT-LM message."""
    role = gemini_content.get("role")
    if role == "model":
        role = "assistant"
    elif not role:
        role = "user"

    parts = gemini_content.get("parts", [])
    litertlm_parts = []
    for p in parts:
        if "text" in p:
            litertlm_parts.append({"type": "text", "text": p["text"]})
    return {"role": role, "content": litertlm_parts}


def litertlm_to_gemini_response(
    litertlm_response: dict[str, Any], finish_reason: str = "STOP"
) -> dict[str, Any]:
    """Converts a LiteRT-LM response to a Gemini API response."""
    parts = []
    for item in litertlm_response.get("content", []):
        if item.get("type") == "text":
            parts.append({"text": item.get("text")})

    candidate: dict[str, Any] = {
        "content": {"role": "model", "parts": parts},
        "index": 0,
    }
    if finish_reason:
        candidate["finishReason"] = finish_reason

    return {"candidates": [candidate]}


class GeminiHandler(http.server.BaseHTTPRequestHandler):
    """Handler for Gemini API requests."""

    # do_POST is the method name expected by http.server.BaseHTTPRequestHandler
    # to handle POST requests.
    def do_POST(self):  # pylint: disable=invalid-name
        """Handles POST requests for generateContent and streamGenerateContent."""
        path_without_query = self.path.split("?")[0]
        gen_match = GEN_CONTENT_RE.match(path_without_query)
        stream_match = STREAM_GEN_CONTENT_RE.match(path_without_query)

        match = gen_match or stream_match
        if not match:
            self.send_error(404, "Not Found")
            return

        model_spec = match.group(1)
        # model_spec can be <model_id>[,<backend>][,<max_tokens>]
    # Support for backend and max_tokens in model_spec is coming soon.
    parts = model_spec.split(",")
    model_id = parts[0]
    backend = parts[1] if len(parts) > 1 else "cpu"

    content_length = int(self.headers.get("Content-Length", 0))
    try:
      body = json.loads(self.rfile.read(content_length))
    except json.JSONDecodeError:
      self.send_error(400, "Invalid JSON")
      return

    try:
      engine = get_engine(model_id, backend)
        except FileNotFoundError as e:
            self.send_error(404, str(e))
            return
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.send_error(500, f"Failed to load engine: {e}")
            return

        system_instruction = None
        si_data = body.get("systemInstruction") or body.get("system_instruction")
        if si_data:
            si_parts = si_data.get("parts", [])
            system_instruction = "".join(p.get("text", "") for p in si_parts)

        messages = [gemini_to_litertlm_message(c) for c in body.get("contents", [])]

        if not messages:
            self.send_error(400, "No contents provided")
            return

        # Last message is the prompt.
        # Note: send_message expects Mapping[str, Any] with 'role' and 'content'.
        last_msg = messages.pop()

        # Prefix messages (context)
        context_messages = []
        if system_instruction:
            context_messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_instruction}],
                }
            )
        context_messages.extend(messages)

        try:
            with engine.create_conversation(messages=context_messages) as conv:
                if stream_match:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()

                    for chunk in conv.send_message_async(last_msg):
                        resp = litertlm_to_gemini_response(chunk, finish_reason="")
                        self.wfile.write(
                            f"data: {json.dumps(resp, ensure_ascii=False)}\n\n".encode(
                                "utf-8"
                            )
                        )
                        self.wfile.flush()

                    # Final chunk to signal completion
                    final_resp = litertlm_to_gemini_response(
                        {"content": []}, finish_reason="STOP"
                    )
                    self.wfile.write(
                        f"data: {json.dumps(final_resp, ensure_ascii=False)}\n\n".encode(
                            "utf-8"
                        )
                    )
                    self.wfile.flush()
                else:
                    response = conv.send_message(last_msg)
                    resp_body = litertlm_to_gemini_response(response)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(resp_body, ensure_ascii=False).encode("utf-8")
                    )

        except Exception as e:  # pylint: disable=broad-exception-caught
            click.echo(click.style(f"Error during inference: {e}", fg="red"))
            if not self.wfile.closed:
                try:
                    self.send_error(500, str(e))
                except BrokenPipeError:
                    pass


def run_server(port: int, verbose: bool):
    """Starts the HTTP server."""
    # The BaseHTTPRequestHandler uses sys.stderr by default for request logs.
    # If verbose is not set, we can quiet down the engine logs.
    if not verbose:
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

    server_address = ("localhost", port)
    httpd = http.server.HTTPServer(server_address, GeminiHandler)
    click.echo(
        click.style(
            f"Starting LiteRT-LM Gemini API server on port {port}...",
            fg="green",
            bold=True,
        )
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        click.echo(click.style("\nShutting down server...", fg="cyan"))
        httpd.server_close()
        if _current_engine:
            _current_engine.__exit__(None, None, None)


def register(cli):
  """Registers the serve command."""

  @cli.command(
      help="Start a server with a Gemini compatible API (alpha feature)"
  )
  @click.option(
      "--port", "-p", default=9379, type=int, help="Port to listen on"
  )
  @click.option("--verbose", is_flag=True, help="Enable verbose logging")
  @click.option(
      "-b",
      "--backend",
      type=click.Choice(["cpu", "gpu"], case_sensitive=False),
      default="cpu",
      help="The backend to use for initialization.",
  )
  def serve(port: int, verbose: bool, backend: str):
    """Starts a local HTTP server speaking the Gemini API protocol.

    Args:
      port: Port to listen on.
      verbose: Whether to enable verbose logging.
      backend: Hardware backend (e.g., cpu or gpu).
    """
    run_server(port, verbose)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI server for the Credit Assessment Environment."""

import os
from typing import Any, Dict, Optional

from fastapi import Body, HTTPException, WebSocket, status
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.websockets import WebSocketDisconnect

try:
    from openenv.core.env_server.http_server import create_app, create_fastapi_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import CreditAssessmentAction, CreditAssessmentObservation
    from .credit_assessment_env_environment import CreditAssessmentEnvironment
except ModuleNotFoundError:
    from models import CreditAssessmentAction, CreditAssessmentObservation
    from server.credit_assessment_env_environment import CreditAssessmentEnvironment


def _web_interface_enabled() -> bool:
    return os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes")


def _create_openenv_app():
    if not _web_interface_enabled():
        return create_app(
            CreditAssessmentEnvironment,
            CreditAssessmentAction,
            CreditAssessmentObservation,
            env_name="credit_assessment_env",
            max_concurrent_envs=1,
        )

    import gradio as gr
    from openenv.core.env_server.web_interface import (
        OPENENV_GRADIO_CSS,
        OPENENV_GRADIO_THEME,
        WebInterfaceManager,
        _extract_action_fields,
        _is_chat_env,
        build_gradio_app,
        get_gradio_display_title,
        get_quick_start_markdown,
        load_environment_metadata,
    )

    app = create_fastapi_app(
        CreditAssessmentEnvironment,
        CreditAssessmentAction,
        CreditAssessmentObservation,
        max_concurrent_envs=1,
    )

    metadata = load_environment_metadata(
        CreditAssessmentEnvironment,
        "credit_assessment_env",
    )
    web_manager = WebInterfaceManager(
        CreditAssessmentEnvironment,
        CreditAssessmentAction,
        CreditAssessmentObservation,
        metadata,
    )

    @app.get("/web/metadata")
    async def web_metadata():
        """Get environment metadata."""
        return web_manager.metadata.model_dump()

    @app.websocket("/ws/ui")
    async def websocket_ui_endpoint(websocket: WebSocket):
        """WebSocket endpoint for web UI real-time updates."""
        await web_manager.connect_websocket(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await web_manager.disconnect_websocket(websocket)

    @app.post("/web/reset")
    async def web_reset(request: Optional[Dict[str, Any]] = Body(default=None)):
        """Reset endpoint for web interface."""
        return await web_manager.reset_environment(request)

    @app.post("/web/step")
    async def web_step(request: Dict[str, Any]):
        """Step endpoint for web interface."""
        if "message" in request:
            message = request["message"]
            if hasattr(web_manager.env, "message_to_action"):
                action = web_manager.env.message_to_action(message)
                if hasattr(action, "tokens"):
                    action_data = {"tokens": action.tokens.tolist()}
                else:
                    action_data = action.model_dump(exclude={"metadata"})
            else:
                action_data = {"message": message}
        else:
            action_data = request.get("action", {})

        return await web_manager.step_environment(action_data)

    @app.get("/web/state")
    async def web_state():
        """State endpoint for web interface."""
        try:
            return web_manager.get_state()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

    @app.get("/playground", include_in_schema=False)
    async def playground_root():
        """Redirect the playground root to the mounted Gradio app."""
        return RedirectResponse(url="/playground/")

    action_fields = _extract_action_fields(CreditAssessmentAction)
    is_chat_env = _is_chat_env(CreditAssessmentAction)
    quick_start_md = get_quick_start_markdown(
        metadata,
        CreditAssessmentAction,
        CreditAssessmentObservation,
    )
    gradio_blocks = build_gradio_app(
        web_manager,
        action_fields,
        metadata,
        is_chat_env,
        title=metadata.name,
        quick_start_md=quick_start_md,
    )
    return gr.mount_gradio_app(
        app,
        gradio_blocks,
        path="/playground",
        theme=OPENENV_GRADIO_THEME,
        css=OPENENV_GRADIO_CSS,
        app_kwargs={"title": get_gradio_display_title(metadata)},
    )


app = _create_openenv_app()


LANDING_PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Credit Assessment Environment</title>
  <meta
    name="description"
    content="A minimal OpenEnv environment where agents learn loan underwriting through synthetic applicants, policy-grounded actions, and reward feedback."
  />
  <style>
    :root {
      --paper: #f7f1e8;
      --paper-deep: #efe4d3;
      --ink: #2b2118;
      --muted: #6f6256;
      --line: #ded1bf;
      --accent: #a84f2f;
      --accent-dark: #74341f;
      --card: rgba(255, 252, 247, 0.74);
      --shadow: 0 24px 80px rgba(67, 48, 31, 0.12);
    }

    * {
      box-sizing: border-box;
    }

    html {
      scroll-behavior: smooth;
    }

    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 15% 10%, rgba(168, 79, 47, 0.14), transparent 28rem),
        radial-gradient(circle at 85% 0%, rgba(117, 84, 51, 0.12), transparent 24rem),
        linear-gradient(180deg, var(--paper) 0%, #fbf7f0 45%, var(--paper-deep) 100%);
      font-family: Georgia, "Times New Roman", serif;
    }

    a {
      color: inherit;
      text-decoration: none;
    }

    .page {
      min-height: 100vh;
      overflow: hidden;
    }

    .nav,
    .hero,
    .section,
    .footer {
      width: min(1120px, calc(100% - 40px));
      margin: 0 auto;
    }

    .nav {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 28px 0 18px;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      letter-spacing: -0.01em;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 650;
    }

    .mark {
      width: 32px;
      height: 32px;
      border: 1px solid var(--ink);
      border-radius: 50%;
      display: grid;
      place-items: center;
      font-family: Georgia, "Times New Roman", serif;
      font-size: 16px;
      line-height: 1;
    }

    .nav-links {
      display: flex;
      align-items: center;
      gap: 22px;
      color: var(--muted);
    }

    .nav-links a:hover {
      color: var(--ink);
    }

    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.08fr) minmax(320px, 0.72fr);
      gap: 44px;
      align-items: center;
      padding: 74px 0 88px;
    }

    .eyebrow {
      width: fit-content;
      margin: 0 0 26px;
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--accent-dark);
      background: rgba(255, 252, 247, 0.54);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 13px;
      font-weight: 650;
      letter-spacing: 0.02em;
    }

    h1 {
      margin: 0;
      max-width: 860px;
      font-size: clamp(54px, 9vw, 112px);
      line-height: 0.92;
      letter-spacing: -0.07em;
      font-weight: 500;
    }

    .lede {
      max-width: 690px;
      margin: 30px 0 0;
      color: var(--muted);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: clamp(18px, 2.2vw, 23px);
      line-height: 1.45;
      letter-spacing: -0.02em;
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 34px;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 48px;
      padding: 0 20px;
      border: 1px solid var(--ink);
      border-radius: 999px;
      font-size: 15px;
      font-weight: 650;
      transition: transform 160ms ease, box-shadow 160ms ease, background 160ms ease;
    }

    .button:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 24px rgba(67, 48, 31, 0.12);
    }

    .button.primary {
      color: #fffaf3;
      background: var(--ink);
    }

    .button.secondary {
      background: rgba(255, 252, 247, 0.46);
    }

    .panel {
      position: relative;
      padding: 28px;
      border: 1px solid var(--line);
      border-radius: 30px;
      background: var(--card);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }

    .panel::before {
      content: "";
      position: absolute;
      inset: 14px;
      z-index: -1;
      border: 1px solid rgba(43, 33, 24, 0.08);
      border-radius: 22px;
    }

    .terminal-label {
      color: var(--muted);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.13em;
      text-transform: uppercase;
    }

    .profile {
      margin: 22px 0;
      padding: 22px;
      border-radius: 20px;
      background: #2b2118;
      color: #f8efe2;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 14px;
      line-height: 1.6;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08);
    }

    .profile span {
      color: #d8a287;
    }

    .score-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .score {
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255, 252, 247, 0.7);
    }

    .score strong {
      display: block;
      margin-bottom: 4px;
      font-size: 22px;
      letter-spacing: -0.04em;
    }

    .score small {
      color: var(--muted);
      font-size: 12px;
    }

    .section {
      padding: 34px 0;
    }

    .section-header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 24px;
    }

    h2 {
      margin: 0;
      max-width: 700px;
      font-size: clamp(34px, 5vw, 64px);
      line-height: 0.98;
      letter-spacing: -0.055em;
      font-weight: 500;
    }

    .section-kicker {
      max-width: 360px;
      margin: 0;
      color: var(--muted);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 16px;
      line-height: 1.5;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
    }

    .card {
      min-height: 220px;
      padding: 24px;
      border: 1px solid var(--line);
      border-radius: 26px;
      background: rgba(255, 252, 247, 0.58);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .card .number {
      color: var(--accent);
      font-size: 13px;
      font-weight: 750;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .card h3 {
      margin: 48px 0 12px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: 27px;
      line-height: 1.02;
      letter-spacing: -0.04em;
      font-weight: 500;
    }

    .card p,
    .wide-card p {
      margin: 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.55;
    }

    .wide-card {
      display: grid;
      grid-template-columns: 0.85fr 1.15fr;
      gap: 28px;
      align-items: center;
      padding: 30px;
      border: 1px solid var(--line);
      border-radius: 30px;
      background: rgba(255, 252, 247, 0.66);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .wide-card h3 {
      margin: 0 0 14px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(30px, 4vw, 48px);
      line-height: 1;
      letter-spacing: -0.05em;
      font-weight: 500;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .chip {
      padding: 10px 13px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--accent-dark);
      background: rgba(247, 241, 232, 0.72);
      font-size: 14px;
      font-weight: 650;
    }

    .playground-shell {
      display: grid;
      grid-template-columns: 0.56fr 1.44fr;
      gap: 18px;
      align-items: stretch;
      min-height: 680px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 34px;
      background: rgba(255, 252, 247, 0.66);
      box-shadow: var(--shadow);
    }

    .playground-copy {
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 24px;
      padding: 22px;
      border-radius: 24px;
      background: #2b2118;
      color: #fbf7f0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .playground-copy h3 {
      margin: 0 0 14px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(30px, 4vw, 48px);
      line-height: 0.98;
      letter-spacing: -0.055em;
      font-weight: 500;
    }

    .playground-copy p {
      margin: 0;
      color: #d9cbbb;
      font-size: 15px;
      line-height: 1.55;
    }

    .playground-steps {
      display: grid;
      gap: 10px;
      margin-top: 22px;
    }

    .playground-step {
      padding: 12px 0;
      border-top: 1px solid rgba(255, 250, 243, 0.16);
      color: #f8efe2;
      font-size: 14px;
      line-height: 1.45;
    }

    .playground-frame-wrap {
      min-height: 640px;
      overflow: hidden;
      border: 1px solid rgba(43, 33, 24, 0.1);
      border-radius: 24px;
      background: #fffaf3;
    }

    .playground-frame {
      width: 100%;
      height: 100%;
      min-height: 640px;
      border: 0;
      background: #fffaf3;
    }

    .footer {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      padding: 54px 0 36px;
      color: var(--muted);
      border-top: 1px solid var(--line);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }

    @media (max-width: 860px) {
      .nav-links {
        display: none;
      }

      .hero,
      .wide-card,
      .playground-shell {
        grid-template-columns: 1fr;
      }

      .hero {
        padding-top: 44px;
      }

      .grid,
      .score-row {
        grid-template-columns: 1fr;
      }

      .section-header,
      .footer {
        align-items: start;
        flex-direction: column;
      }

      .playground-shell {
        min-height: auto;
      }

      .playground-frame-wrap,
      .playground-frame {
        min-height: 760px;
      }
    }
  </style>
</head>
<body>
  <main class="page">
    <nav class="nav" aria-label="Primary navigation">
      <a class="brand" href="/web" aria-label="Credit Assessment Environment home">
        <span class="mark">C</span>
        <span>Credit Assessment Env</span>
      </a>
      <div class="nav-links">
        <a href="#how-it-works">How it works</a>
        <a href="#observations">Observations</a>
        <a href="#playground">Playground</a>
        <a href="/docs">API docs</a>
      </div>
    </nav>

    <section class="hero">
      <div>
        <p class="eyebrow">OpenEnv for policy-grounded credit decisions</p>
        <h1>Teach an agent to reason like a careful loan officer.</h1>
        <p class="lede">
          A compact reinforcement-learning environment for credit assessment. Agents read synthetic loan files, choose underwriting actions, and receive rewards shaped by RBI-style lending rules.
        </p>
        <div class="actions" aria-label="Primary actions">
          <a class="button primary" href="#playground">Try the Playground</a>
          <a class="button secondary" href="#how-it-works">See the episode flow</a>
        </div>
      </div>

      <aside class="panel" aria-label="Example applicant observation">
        <div class="terminal-label">Observation sample</div>
        <div class="profile">
          loan_type: <span>vehicle</span><br />
          cibil_score: <span>718</span><br />
          foir: <span>43%</span><br />
          ltv_ratio: <span>88%</span><br />
          documents: <span>complete</span><br />
          action_needed: <span>counter_offer</span>
        </div>
        <div class="score-row">
          <div class="score">
            <strong>4</strong>
            <small>actions</small>
          </div>
          <div class="score">
            <strong>3</strong>
            <small>loan types</small>
          </div>
          <div class="score">
            <strong>1</strong>
            <small>reward signal</small>
          </div>
        </div>
      </aside>
    </section>

    <section class="section" id="how-it-works">
      <div class="section-header">
        <h2>A small environment with real underwriting pressure.</h2>
        <p class="section-kicker">
          The task is not to sound confident. It is to make the right call when CIBIL, FOIR, documents, LTV, and RERA constraints pull in different directions.
        </p>
      </div>
      <div class="grid">
        <article class="card">
          <div class="number">Step 01</div>
          <h3>Reset</h3>
          <p>Start a fresh episode and receive a narrative applicant profile with structured credit fields.</p>
        </article>
        <article class="card">
          <div class="number">Step 02</div>
          <h3>Act</h3>
          <p>Choose approve, reject, request documents, or counter-offer with a reasoned explanation.</p>
        </article>
        <article class="card">
          <div class="number">Step 03</div>
          <h3>Learn</h3>
          <p>The environment scores the action against policy logic, penalizing risky approvals and missed good loans.</p>
        </article>
      </div>
    </section>

    <section class="section" id="observations">
      <div class="wide-card">
        <div>
          <h3>What the agent sees</h3>
          <p>
            Each observation reads like a loan file, then exposes the fields needed for exact reasoning. Personal loans test core eligibility. Vehicle loans add LTV discipline. Home loans add tiered LTV, RERA compliance, and co-applicant nuance.
          </p>
        </div>
        <div class="chips" aria-label="Observation fields">
          <span class="chip">Applicant profile</span>
          <span class="chip">CIBIL score</span>
          <span class="chip">FOIR</span>
          <span class="chip">Employment years</span>
          <span class="chip">Loan amount</span>
          <span class="chip">Documents status</span>
          <span class="chip">Collateral value</span>
          <span class="chip">LTV ratio</span>
          <span class="chip">RERA registration</span>
        </div>
      </div>
    </section>

    <section class="section" id="playground">
      <div class="section-header">
        <h2>Try the environment from the homepage.</h2>
        <p class="section-kicker">
          Reset to draw a synthetic applicant, enter an underwriting decision, then inspect the reward and next observation without leaving the landing page.
        </p>
      </div>
      <div class="playground-shell">
        <aside class="playground-copy">
          <div>
            <div class="terminal-label">Live playground</div>
            <h3>Make a decision. See the feedback.</h3>
            <p>
              The embedded playground is the original OpenEnv interface. Use it to test approve, reject, request_docs, and counter_offer flows against the same API used by agents.
            </p>
            <div class="playground-steps">
              <div class="playground-step">1. Click Reset to create a new loan file.</div>
              <div class="playground-step">2. Choose an action and provide reasoning.</div>
              <div class="playground-step">3. Step the environment and inspect reward, done, and raw JSON.</div>
            </div>
          </div>
          <a class="button secondary" href="/playground/" target="_blank" rel="noreferrer">Open full playground</a>
        </aside>
        <div class="playground-frame-wrap">
          <iframe
            class="playground-frame"
            title="Credit Assessment Environment Playground"
            src="/playground/"
            loading="lazy"
          ></iframe>
        </div>
      </div>
    </section>

    <footer class="footer">
      <span>Built with OpenEnv for synthetic, auditable credit-assessment training.</span>
      <span><a href="/playground/">Playground</a> / <a href="/health">Health</a> / <a href="/schema">Schema</a> / <a href="/metadata">Metadata</a></span>
    </footer>
  </main>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
@app.get("/web/", response_class=HTMLResponse, include_in_schema=False)
def landing_page() -> HTMLResponse:
    """Serve the Hugging Face Space landing page without affecting API routes."""
    return HTMLResponse(content=LANDING_PAGE_HTML)


def main(host: str = "0.0.0.0", port: int = 7860):
    """Run the server directly: uv run --project . server"""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

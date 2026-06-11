"""
orchestrator.py — 2x2 TUI dashboard, terminator-style layout.

Keybindings:
  f  — skip fetcher, jump straight to NLP
  q  — quit
"""
import os
import asyncio
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Log

BASE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = {
    "Tweet Fetcher": ["python", os.path.join(BASE, "nlp_pipeline", "initial_fetch.py")],
    "NLP Processor": ["python", os.path.join(BASE, "nlp_pipeline", "NLProcessing.py")],
    "Node Server":   ["node",   os.path.join(BASE, "backend", "server.js")],
    "Frontend":      ["python", "-m", "http.server", "8080",
                      "--directory", os.path.join(BASE, "frontend", "public")],
}

FETCHER_ERROR_PHRASES = ["402", "Payment Required", "credits", "Unauthorized", "401", "ERROR"]

PANEL_COLORS = {
    "Tweet Fetcher": "#00aaff",   # blue
    "NLP Processor": "#00cc66",   # green
    "Node Server":   "#ff9900",   # amber
    "Frontend":      "#cc44ff",   # purple
}

def to_id(name: str) -> str:
    return name.replace(' ', '-').lower()


class Panel(Vertical):
    def __init__(self, script_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.script_name = script_name
        self.pid = to_id(script_name)

    def compose(self) -> ComposeResult:
        yield Static(
            f" ● {self.script_name.upper()}  [dim]waiting[/dim]",
            id=f"s-{self.pid}",
            classes="bar"
        )
        yield Log(id=f"l-{self.pid}", auto_scroll=True)

    def set_status(self, state: str):
        icons = {
            "running": ("●", "green",  "RUNNING"),
            "stopped": ("■", "red",    "STOPPED"),
            "skipped": ("▶", "yellow", "SKIPPED"),
            "error":   ("▲", "red",    "ERROR"),
            "waiting": ("○", "white",  "WAITING"),
        }
        dot, color, label = icons.get(state, ("○", "white", "WAITING"))
        self.query_one(f"#s-{self.pid}", Static).update(
            f" [{color}]{dot}[/{color}] {self.script_name.upper()}  [dim]{label}[/dim]"
        )

    def log(self, line: str):
        w = self.query_one(f"#l-{self.pid}", Log)
        w.write_line(line)


class Orchestrator(App):
    BINDINGS = [
        Binding("f", "skip_fetcher", "Skip Fetcher", show=True),
        Binding("q", "quit",         "Quit",         show=True),
    ]

    CSS = f"""
    Screen {{
        background: #0a0a0a;
    }}

    Header {{
        background: #111111;
        color: #888888;
        height: 1;
        dock: top;
    }}

    Footer {{
        background: #111111;
        color: #555555;
        height: 1;
        dock: bottom;
    }}

    /* 2x2 grid */
    #row-top, #row-bot {{
        height: 1fr;
    }}

    Panel {{
        border: solid #2a2a2a;
        margin: 0;
        padding: 0;
    }}

    /* Per-panel accent colors on the title bar */
    #tweet-fetcher .bar {{ background: #001a2e; color: {PANEL_COLORS["Tweet Fetcher"]}; }}
    #nlp-processor .bar {{ background: #001a0f; color: {PANEL_COLORS["NLP Processor"]}; }}
    #node-server   .bar {{ background: #1a1000; color: {PANEL_COLORS["Node Server"]};   }}
    #frontend      .bar {{ background: #110022; color: {PANEL_COLORS["Frontend"]};      }}

    .bar {{
        height: 1;
        padding: 0 1;
        text-style: bold;
    }}

    Log {{
        height: 1fr;
        background: #0d0d0d;
        color: #aaaaaa;
        scrollbar-size: 1 1;
        scrollbar-color: #333333;
        text-style: none;
        padding: 0 1;
    }}

    /* Small text via padding trick — Textual doesn't support font-size
       but keeping padding tight gives a dense feel */
    """

    def __init__(self):
        super().__init__()
        self.nlp_ready   = asyncio.Event()
        self.node_ready  = asyncio.Event()
        self._fetcher_proc = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            with Horizontal(id="row-top"):
                yield Panel("Tweet Fetcher", id="tweet-fetcher")
                yield Panel("NLP Processor", id="nlp-processor")
            with Horizontal(id="row-bot"):
                yield Panel("Node Server",   id="node-server")
                yield Panel("Frontend",      id="frontend")
        yield Footer()

    async def on_mount(self):
        asyncio.create_task(self.run_fetcher())

        asyncio.create_task(self.run_script(
            "NLP Processor", SCRIPTS["NLP Processor"],
            wait=self.nlp_ready,
            trigger=self.node_ready, trigger_phrase="Sleeping 30s"))

        asyncio.create_task(self.run_script(
            "Node Server", SCRIPTS["Node Server"],
            wait=self.node_ready))

        asyncio.create_task(self.run_script(
            "Frontend", SCRIPTS["Frontend"]))

    # ── Fetcher with auto-bypass ──────────────────────────────────────────────
    async def run_fetcher(self):
        panel = self.query_one("#tweet-fetcher", Panel)
        panel.set_status("running")

        proc = await asyncio.create_subprocess_exec(
            *SCRIPTS["Tweet Fetcher"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        self._fetcher_proc = proc
        auto_bypassed = False

        async for raw in proc.stdout:
            line = raw.decode(errors="ignore").rstrip()
            panel.log(line)
            if not auto_bypassed and any(p in line for p in FETCHER_ERROR_PHRASES):
                panel.log("⚠  API error — auto-skipping to NLP")
                panel.set_status("error")
                auto_bypassed = True
                self.nlp_ready.set()
                self.node_ready.set()

        await proc.wait()
        if not auto_bypassed:
            panel.set_status("stopped")
            self.nlp_ready.set()
            self.node_ready.set()

    # ── Manual bypass: F key ──────────────────────────────────────────────────
    def action_skip_fetcher(self):
        panel = self.query_one("#tweet-fetcher", Panel)
        if self.nlp_ready.is_set():
            panel.log("ℹ  Already past fetcher stage")
            return
        if self._fetcher_proc and self._fetcher_proc.returncode is None:
            self._fetcher_proc.terminate()
        panel.log("▶  Manually skipped — using existing Supabase data")
        panel.set_status("skipped")
        self.nlp_ready.set()
        self.node_ready.set()

    # ── Generic runner ────────────────────────────────────────────────────────
    async def run_script(self, name, cmd, wait=None, trigger=None, trigger_phrase=None):
        panel = self.query_one(f"#{to_id(name)}", Panel)
        if wait:
            await wait.wait()

        panel.set_status("running")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )

        async for raw in proc.stdout:
            line = raw.decode(errors="ignore").rstrip()
            panel.log(line)
            if trigger and trigger_phrase and trigger_phrase in line:
                trigger.set()

        await proc.wait()
        panel.set_status("stopped")


if __name__ == "__main__":
    Orchestrator().run()
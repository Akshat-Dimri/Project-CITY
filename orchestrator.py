"""
orchestrator.py — starts all four project components in a TUI dashboard.
Uses Textual for the terminal UI.
"""
import os
import asyncio
from textual.app import App, ComposeResult
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

def to_id(name: str) -> str:
    return name.replace(' ', '-').lower()


class Panel(Vertical):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.pid  = to_id(name)

    def compose(self) -> ComposeResult:
        yield Static(f"[⚪] {self.name}", classes="status", id=f"s-{self.pid}")
        yield Log(id=f"l-{self.pid}")

    def on_mount(self):
        self.query_one(f"#l-{self.pid}", Log).write_line("waiting...")

    def set_status(self, state: str):
        icon = {"running": "🟢", "stopped": "🔴"}.get(state, "⚪")
        self.query_one(f"#s-{self.pid}", Static).update(f"[{icon}] {self.name}")

    def log(self, line: str):
        w = self.query_one(f"#l-{self.pid}", Log)
        if w.line_count == 1 and "waiting" in w._lines[0]:
            w.clear()
        w.write_line(line)


class Orchestrator(App):
    CSS = """
    .status { padding: 1; background: $boost; color: yellow; content-align: center middle; }
    Log { height: 1fr; }
    Horizontal > Panel { width: 1fr; }
    """

    def __init__(self):
        super().__init__()
        self.nlp_ready  = asyncio.Event()
        self.node_ready = asyncio.Event()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            for name in SCRIPTS:
                yield Panel(name, id=to_id(name))
        yield Footer()

    async def on_mount(self):
        asyncio.create_task(self.run_script(
            "Tweet Fetcher", SCRIPTS["Tweet Fetcher"],
            trigger=self.nlp_ready, trigger_phrase="Rate limit"))

        asyncio.create_task(self.run_script(
            "NLP Processor", SCRIPTS["NLP Processor"],
            wait=self.nlp_ready,
            trigger=self.node_ready, trigger_phrase="Sleeping 30s"))

        asyncio.create_task(self.run_script(
            "Node Server", SCRIPTS["Node Server"],
            wait=self.node_ready))

        asyncio.create_task(self.run_script(
            "Frontend", SCRIPTS["Frontend"]))

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

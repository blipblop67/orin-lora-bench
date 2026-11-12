"""Start / health-check / stop llama-server and vLLM, in docker or natively."""
from __future__ import annotations

import json
import os
import subprocess
import time

import requests

ROOT = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


class Server:
    def __init__(self, name: str, port: int, argv: list[str], image: str | None = None,
                 native_bin: str | None = None, ipc_host: bool = False):
        self.name, self.port, self.argv = name, port, argv
        self.image, self.native_bin, self.ipc_host = image, native_bin, ipc_host
        self.proc = None
        self.logfile = os.path.join(ROOT, "outputs", "logs", f"{name}.log")
        os.makedirs(os.path.dirname(self.logfile), exist_ok=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        self.stop()
        if self.native_bin:
            self._log = open(self.logfile, "w")
            self.proc = subprocess.Popen([self.native_bin, *self.argv], stdout=self._log, stderr=subprocess.STDOUT)
        else:
            hf = os.environ.get("HF_CACHE", os.path.expanduser("~/.cache/huggingface"))
            cmd = ["docker", "run", "-d", "--name", self.name, "--runtime", "nvidia", "--network", "host",
                   "-v", f"{ROOT}:{ROOT}", "-v", f"{hf}:/root/.cache/huggingface", "-w", ROOT]
            if self.ipc_host:
                cmd += ["--ipc", "host"]
            if os.environ.get("HF_TOKEN"):
                cmd += ["-e", "HF_TOKEN"]
            r = _run(cmd + [self.image, *self.argv])
            if r.returncode != 0:
                raise RuntimeError(f"docker run failed: {r.stderr.strip()}")
        self.t_start = time.time()

    def alive(self) -> bool:
        if self.native_bin:
            return self.proc is not None and self.proc.poll() is None
        r = _run(["docker", "inspect", "-f", "{{.State.Running}}", self.name])
        return r.stdout.strip() == "true"

    def exit_info(self) -> dict:
        if self.native_bin:
            return {"exit_code": self.proc.poll() if self.proc else None}
        r = _run(["docker", "inspect", "-f", "{{json .State}}", self.name])
        try:
            s = json.loads(r.stdout)
            return {"exit_code": s.get("ExitCode"), "oom_killed": s.get("OOMKilled")}
        except json.JSONDecodeError:
            return {}

    def wait_ready(self, timeout: float = 300) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            try:
                if requests.get(self.url + "/health", timeout=2).status_code == 200:
                    self.startup_s = round(time.time() - self.t_start, 1)
                    return True
            except requests.RequestException:
                pass
            if not self.alive():
                return False
            time.sleep(1)
        return False

    def logs(self, tail: int | None = None) -> str:
        if self.native_bin:
            try:
                txt = open(self.logfile).read()
            except OSError:
                return ""
        else:
            r = _run(["docker", "logs", self.name])
            txt = r.stdout + r.stderr
            with open(self.logfile, "w") as f:
                f.write(txt)
        return "\n".join(txt.splitlines()[-tail:]) if tail else txt

    def stop(self):
        if self.native_bin:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(30)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        else:
            _run(["docker", "rm", "-f", self.name])
        time.sleep(2)  # let unified memory be reclaimed before the next measurement


def llama_server(argv: list[str], port: int) -> Server:
    bindir = os.environ.get("LLAMA_CPP_BIN_DIR") or None
    return Server("lcpp-bench", port, ["llama-server", *argv] if not bindir else argv,
                  image=os.environ.get("LLAMA_CPP_IMAGE"),
                  native_bin=os.path.join(bindir, "llama-server") if bindir else None)


def vllm_server(argv: list[str], port: int) -> Server:
    return Server("vllm-bench", port, ["vllm", "serve", *argv], image=os.environ.get("VLLM_IMAGE"), ipc_host=True)


_FA_STYLE = None


def llama_fa_args() -> list[str]:
    """llama-server's flash-attention flag changed from a bare switch to '--flash-attn on|off|auto'."""
    global _FA_STYLE
    if _FA_STYLE is None:
        bindir = os.environ.get("LLAMA_CPP_BIN_DIR")
        cmd = [os.path.join(bindir, "llama-server")] if bindir else \
            ["docker", "run", "--rm", "--runtime", "nvidia", os.environ["LLAMA_CPP_IMAGE"], "llama-server"]
        r = _run(cmd + ["--help"])
        h = r.stdout + r.stderr
        _FA_STYLE = "value" if "[on|off|auto]" in h else "switch"
    return ["--flash-attn", "on"] if _FA_STYLE == "value" else ["--flash-attn"]

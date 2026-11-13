"""Memory / GPU sampling for Jetson unified memory.

On Jetson the CPU and GPU share DRAM, so "does it fit" is a system-wide question:
we track used = MemTotal - MemAvailable from /proc/meminfo (CUDA allocations show
up there), swap use, and GR3D (GPU) load from tegrastats when available.

CLI:
  python memmon.py --snapshot            # print current used MB
  python memmon.py --device-meta         # JSON device / JetPack / power-mode info
  python memmon.py --out m.json -- cmd   # run cmd and record memory while it runs
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time


def meminfo() -> dict:
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            d[k] = int(v.split()[0]) // 1024  # MB
    return d


def used_mb() -> int:
    m = meminfo()
    return m["MemTotal"] - m["MemAvailable"]


def swap_used_mb() -> int:
    m = meminfo()
    return m.get("SwapTotal", 0) - m.get("SwapFree", 0)


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def power_mode() -> str:
    out = _sh(["nvpmodel", "-q"])
    m = re.search(r"NV Power Mode:\s*(\S+)", out)
    return m.group(1) if m else "unknown"


def device_meta() -> dict:
    def read(p):
        try:
            with open(p) as f:
                return f.read().replace("\x00", "").strip()
        except OSError:
            return ""
    return {
        "model": read("/proc/device-tree/model"),
        "l4t": read("/etc/nv_tegra_release").splitlines()[0] if read("/etc/nv_tegra_release") else "",
        "kernel": os.uname().release,
        "mem_total_mb": meminfo()["MemTotal"],
        "swap_total_mb": meminfo().get("SwapTotal", 0),
        "power_mode": power_mode(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def image_digest(image: str | None) -> str | None:
    """Pinned digest of a local docker image (tags like :latest move; the digest says what actually ran)."""
    if not image:
        return None
    out = _sh(["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", image])
    return out or None


_DROP_OK: bool | None = None


def drop_caches() -> bool:
    """sync + drop page/dentry/inode caches between CUDA workloads.

    On JetPack < 7.2 (L4T < r39) memory freed by an exited CUDA process can stay in caches and look
    used, which skews the next config's baseline and fit result. Needs root or passwordless sudo.
    DROP_CACHES=0 off, 1 on, auto (default) on when `sudo -n` works, otherwise warn once and skip.
    """
    global _DROP_OK
    mode = os.environ.get("DROP_CACHES", "auto")
    if mode == "0" or _DROP_OK is False:
        return False
    cmd = "sync && echo 3 | tee /proc/sys/vm/drop_caches >/dev/null"
    if os.geteuid() != 0:
        cmd = "sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null"
    ok = subprocess.run(cmd, shell=True, check=False, stderr=subprocess.DEVNULL).returncode == 0
    if not ok and _DROP_OK is None:
        print("[warn] drop_caches needs passwordless sudo; skipping. Baselines between configs may be "
              "inflated by cached memory (see README > Troubleshooting).", file=sys.stderr, flush=True)
    _DROP_OK = ok
    return ok


class MemMonitor:
    """Background sampler. Use as a context manager; read .summary() afterwards."""

    def __init__(self, interval: float = 0.5, baseline_mb: int | None = None):
        self.interval = interval
        self.baseline_mb = used_mb() if baseline_mb is None else baseline_mb
        self.samples: list[tuple[float, int, int]] = []
        self.gr3d: list[int] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._teg = None

    def _run(self):
        while not self._stop.is_set():
            self.samples.append((time.time(), used_mb(), swap_used_mb()))
            self._stop.wait(self.interval)

    def _tegra_reader(self):
        for line in self._teg.stdout:
            m = re.search(r"GR3D_FREQ (\d+)%", line)
            if m:
                self.gr3d.append(int(m.group(1)))

    def __enter__(self):
        self._t.start()
        if shutil.which("tegrastats"):
            try:
                self._teg = subprocess.Popen(["tegrastats", "--interval", str(int(self.interval * 1000))],
                                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                threading.Thread(target=self._tegra_reader, daemon=True).start()
            except OSError:
                self._teg = None
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        if self._teg:
            self._teg.terminate()
        return False

    def summary(self) -> dict:
        if not self.samples:
            return {}
        used = [s[1] for s in self.samples]
        swap = [s[2] for s in self.samples]
        return {
            "mem_baseline_mb": self.baseline_mb,
            "mem_peak_used_mb": max(used),
            "mem_peak_delta_mb": max(used) - self.baseline_mb,
            "swap_peak_used_mb": max(swap),
            "gpu_util_avg_pct": round(sum(self.gr3d) / len(self.gr3d), 1) if self.gr3d else None,
            "mem_samples": len(used),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--device-meta", action="store_true")
    ap.add_argument("--drop-caches", action="store_true", help="drop caches per DROP_CACHES, print used MB")
    ap.add_argument("--out")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.snapshot:
        print(used_mb())
        return
    if a.device_meta:
        meta = device_meta()
        meta["images"] = {k: image_digest(os.environ.get(k)) for k in ("LLAMA_CPP_IMAGE", "VLLM_IMAGE")}
        print(json.dumps(meta, indent=2))
        return
    if a.drop_caches:
        drop_caches()
        print(used_mb())
        return
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("nothing to run")
    with MemMonitor() as mon:
        rc = subprocess.call(cmd)
    s = mon.summary() | {"cmd": cmd, "returncode": rc}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(s, f, indent=2)
    print(json.dumps(s))
    sys.exit(rc)


if __name__ == "__main__":
    main()

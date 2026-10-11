"""``potato probe`` — detect and record the hardware environment.

Outputs a ``hardware.json`` (project-spec section 7 CLI contract:
``potato probe -> hardware.json``) and feeds the environment fingerprint in
``docs/memory.md``.

Design constraints (docs/rules.md):
  * Standard library only — no psutil / GPUtil / pyadl. We shell out to the
    vendor tools that are already present (nvidia-smi, vulkaninfo) and read
    OS-native sources (WMI on Windows, /proc on Linux).
  * Never invent a number. If a field cannot be detected, it is recorded as
    ``null`` with a note, so the JSON is self-describing about what is unknown.
    Detection happens on the user's machine; this module only *describes* how.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

PROBE_SCHEMA_VERSION = 1


def _run(cmd: List[str], timeout: int = 20) -> Optional[str]:
    """Run a command, return stdout on success, else None (never raises)."""
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, *cmd[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        # Some tools write useful data to stderr even on nonzero exit.
        return proc.stdout or proc.stderr or None
    return proc.stdout


def _detect_os() -> Dict[str, Any]:
    system = platform.system()
    info: Dict[str, Any] = {"system": system, "release": platform.release()}
    info["platform"] = {"Windows": "Windows", "Linux": "Linux"}.get(system, system)
    return info


def _cpu_count_logical() -> Optional[int]:
    import os

    return os.cpu_count()


def _cpu_count_physical() -> Optional[int]:
    # Best-effort; the standard library has no portable physical-core API.
    if platform.system() == "Windows":
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).NumberOfCores",
            ]
        )
        if out:
            m = re.search(r"\d+", out)
            if m:
                return int(m.group())
    return None


def _detect_cpu() -> Dict[str, Any]:
    model = platform.processor() or None
    if platform.system() == "Windows":
        # platform.processor() returns an arch string on Windows; prefer the
        # friendly marketing name from WMI for the fingerprint.
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).Name",
            ]
        )
        if out:
            name = out.strip().splitlines()[0].strip()
            if name:
                model = name
    return {
        "model": model,
        "cores_logical": _cpu_count_logical(),
        "cores_physical": _cpu_count_physical(),
    }


def _detect_nvidia() -> Dict[str, Any]:
    """Query nvidia-smi for GPU name, driver, VRAM, and compute capability."""
    info: Dict[str, Any] = {"present": False}
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    )
    if not out:
        return info
    info["present"] = True
    parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
    if len(parts) >= 1 and parts[0]:
        info["name"] = parts[0]
    if len(parts) >= 2 and parts[1]:
        info["driver"] = parts[1]
    if len(parts) >= 3 and parts[2]:
        info["vram"] = parts[2]
    if len(parts) >= 4 and parts[3]:
        cc = parts[3].strip()
        info["compute_cap"] = cc
        # SM major.minor -> CMAKE_CUDA_ARCHITECTURES value (e.g. "7.5" -> "75").
        m = re.match(r"(\d+)\.(\d+)", cc)
        if m:
            info["cuda_arch"] = f"{m.group(1)}{m.group(2)}"
    return info


def _detect_vulkan() -> Dict[str, Any]:
    info: Dict[str, Any] = {"runtime_present": shutil.which("vulkaninfo") is not None}
    out = _run(["vulkaninfo", "--summary"])
    if out:
        info["summary_available"] = True
        m = re.search(r"deviceName\s*=\s*(.+)", out)
        if m:
            info["device"] = m.group(1).strip()
    return info


def _detect_cuda_toolkit() -> Dict[str, Any]:
    info: Dict[str, Any] = {"nvcc_present": shutil.which("nvcc") is not None}
    if info["nvcc_present"]:
        out = _run(["nvcc", "--version"])
        if out:
            m = re.search(r"release\s+(\d+\.\d+)", out)
            if m:
                info["version"] = m.group(1)
    return info


def _detect_ram_gb() -> Optional[float]:
    if platform.system() == "Windows":
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)",
            ]
        )
        if out:
            m = re.search(r"(\d+(?:\.\d+)?)", out)
            if m:
                return float(m.group(1))
    else:  # Linux
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        match = re.search(r"\d+", line)
                        if match:
                            return round(int(match.group()) / (1024 * 1024), 1)
        except OSError:
            pass
    return None


def probe() -> Dict[str, Any]:
    """Run all detectors and return the hardware.json payload.

    Every unknown is ``None`` (never a fabricated value). The ``notes`` list
    records actionable gaps (e.g. CUDA toolkit missing) so the record is
    self-describing about what is unknown.
    """
    notes: List[str] = []

    os_info = _detect_os()
    cpu_info = _detect_cpu()
    nvidia = _detect_nvidia()
    vulkan = _detect_vulkan()
    cuda = _detect_cuda_toolkit()
    ram_gb = _detect_ram_gb()

    if nvidia.get("present"):
        if not nvidia.get("cuda_arch"):
            notes.append(
                "Could not read compute_cap from nvidia-smi; set CUDA arch manually."
            )
    else:
        notes.append("No NVIDIA GPU detected via nvidia-smi.")

    if not cuda.get("nvcc_present"):
        notes.append(
            "CUDA Toolkit (nvcc) not on PATH — a CUDA build is not possible until it "
            "is installed. Vulkan and CPU builds do not need nvcc."
        )

    if vulkan.get("runtime_present") and "device" not in vulkan:
        notes.append("Vulkan runtime present but device name not parsed.")

    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "os": os_info,
        "cpu": cpu_info,
        "ram_gb": ram_gb,
        "gpu": {"vendor": "NVIDIA" if nvidia.get("present") else None, "nvidia": nvidia},
        "vulkan": vulkan,
        "cuda_toolkit": cuda,
        "python": {"version": platform.python_version()},
        "notes": notes,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry: ``python -m potato.probe [--out PATH]`` (default hardware.json)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    out_path = "hardware.json"
    if "--out" in argv:
        i = argv.index("--out")
        try:
            out_path = argv[i + 1]
        except IndexError:
            print("--out requires a path", file=sys.stderr)
            return 2
        del argv[i : i + 2]

    payload = probe()
    text = json.dumps(payload, indent=2, sort_keys=False)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"Wrote {out_path}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

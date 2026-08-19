from __future__ import annotations

"""Safe launcher for user-configured external image-processing commands.

Historically LaMa/SR templates were always executed through ``shell=True``.
That was convenient for pipelines containing ``|``/``&&``, but it also meant a
config file was effectively executable shell code.  The default path is now
argv-based and shell-free.  Advanced users can explicitly opt back into shell
syntax with the corresponding config flag.
"""

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ExternalCommandResult:
    rendered: str
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    shell: bool


def render_command(template: str, values: Mapping[str, object]) -> str:
    escaped = {str(k): shlex.quote(str(v)) for k, v in values.items()}
    try:
        return str(template).format(**escaped)
    except KeyError as exc:
        raise ValueError(f"外部命令模板包含未知占位符：{exc.args[0]}") from exc


def run_external_command(
    template: str,
    values: Mapping[str, object],
    *,
    timeout: int | float,
    allow_shell: bool = False,
) -> ExternalCommandResult:
    rendered = render_command(template, values)
    if allow_shell:
        proc = subprocess.run(
            rendered,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        argv: tuple[str, ...] = (rendered,)
    else:
        try:
            parsed = tuple(shlex.split(rendered, posix=True))
        except ValueError as exc:
            raise ValueError(f"外部命令无法安全解析为 argv：{exc}") from exc
        if not parsed:
            raise ValueError("外部命令为空。")
        proc = subprocess.run(
            list(parsed),
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        argv = parsed
    return ExternalCommandResult(
        rendered=rendered,
        argv=argv,
        returncode=int(proc.returncode),
        stdout=str(proc.stdout or ""),
        stderr=str(proc.stderr or ""),
        shell=bool(allow_shell),
    )

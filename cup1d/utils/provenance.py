"""Code versions recorded with saved results."""

import importlib.util
import subprocess
from pathlib import Path

PACKAGES = ("cup1d", "lace", "forestflow", "lya_interface")


def code_versions(packages=PACKAGES):
    """Git commit of each installed package, and whether it had local changes.

    Parameters
    ----------
    packages : sequence of str
        Importable package names; those not installed are skipped.

    Returns
    -------
    dict
        ``{package: {"commit": sha or None, "dirty": bool or None}}``; ``None``
        when the package is not in a git checkout.
    """
    versions = {}
    for name in packages:
        spec = importlib.util.find_spec(name)  # location only, no import
        if spec is None or spec.origin is None:
            continue
        folder = Path(spec.origin).resolve().parent
        versions[name] = {"commit": _git(folder, "rev-parse", "HEAD"), "dirty": None}
        if versions[name]["commit"] is not None:
            status = _git(folder, "status", "--porcelain", "--untracked-files=no")
            versions[name]["dirty"] = bool(status)
    return versions


def _git(folder, *command):
    """Output of a git command run in ``folder``, or ``None`` if it fails."""
    try:
        result = subprocess.run(["git", "-C", str(folder), *command], capture_output=True,
                                text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()

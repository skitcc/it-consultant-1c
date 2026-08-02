"""Integration tests for deploy/install.sh against a fake root (no host systemd)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO_ROOT / "deploy" / "install.sh"


def _run_install(dest: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    cmd = [str(INSTALL_SH), "--dest-dir", str(dest), *extra]
    return subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    return root


def test_layout_only_install_creates_expected_tree(fake_root: Path) -> None:
    result = _run_install(fake_root, "--layout-only")
    assert "layout-only" in result.stdout
    assert "skipping systemctl" in result.stdout

    assert (fake_root / "opt" / "it-consultant").is_dir()
    assert (fake_root / "var" / "lib" / "it-consultant" / "db").is_dir()
    env_file = fake_root / "etc" / "it-consultant" / ".env"
    assert env_file.is_file()
    env_text = env_file.read_text(encoding="utf-8")
    assert "WATCH_PATH=/var/lib/it-consultant/db" in env_text
    assert "EWS_SERVER=" in env_text

    units = fake_root / "etc" / "systemd" / "system"
    assert (units / "reindex.service").is_file()
    assert (units / "mail-gateway.service").is_file()
    assert (units / "it-consultant.target").is_file()

    # Must not create a real host venv path or touch /etc outside fake root.
    assert not (fake_root / "opt" / "it-consultant" / ".venv").exists()


def test_unit_files_point_at_venv_and_etc(fake_root: Path) -> None:
    _run_install(fake_root, "--layout-only")
    reindex = (fake_root / "etc" / "systemd" / "system" / "reindex.service").read_text(
        encoding="utf-8"
    )
    mail = (fake_root / "etc" / "systemd" / "system" / "mail-gateway.service").read_text(
        encoding="utf-8"
    )
    target = (fake_root / "etc" / "systemd" / "system" / "it-consultant.target").read_text(
        encoding="utf-8"
    )

    assert "ExecStart=/opt/it-consultant/.venv/bin/python -m reindex" in reindex
    assert "EnvironmentFile=-/etc/it-consultant/.env" in reindex
    assert "User=it-consultant" in reindex

    assert "ExecStart=/opt/it-consultant/.venv/bin/python -m mail_gateway" in mail
    assert "Also=mail-gateway.service reindex.service" in target
    assert "WantedBy=multi-user.target" in target


def test_layout_only_preserves_existing_env(fake_root: Path) -> None:
    env_path = fake_root / "etc" / "it-consultant" / ".env"
    env_path.parent.mkdir(parents=True)
    env_path.write_text("EWS_SERVER=keep-me\nWATCH_PATH=/custom\n", encoding="utf-8")
    _run_install(fake_root, "--layout-only")
    assert "EWS_SERVER=keep-me" in env_path.read_text(encoding="utf-8")


def test_install_does_not_invoke_systemctl(fake_root: Path) -> None:
    result = _run_install(fake_root, "--layout-only", "--enable")
    assert "skipping systemctl" in result.stdout


def _can_create_venv(tmp_path: Path) -> bool:
    venv_dir = tmp_path / "probe-venv"
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    ok = result.returncode == 0 and (venv_dir / "bin" / "python").exists()
    if venv_dir.exists():
        shutil.rmtree(venv_dir, ignore_errors=True)
    return ok


@pytest.mark.slow
def test_full_install_venv_and_smoke_reindex(fake_root: Path, tmp_path: Path) -> None:
    if not _can_create_venv(tmp_path):
        pytest.skip("python venv/ensurepip not available on this host")

    _run_install(fake_root)

    venv_python = fake_root / "opt" / "it-consultant" / ".venv" / "bin" / "python"
    assert venv_python.is_file()

    # Smoke: packages importable from the installed venv.
    subprocess.run(
        [str(venv_python), "-c", "import common, mail_gateway, reindex, watchdog"],
        check=True,
        capture_output=True,
        text=True,
    )

    watch = fake_root / "var" / "lib" / "it-consultant" / "db"
    env = os.environ.copy()
    env.update(
        {
            "EWS_SERVER": "mail.example.com",
            "EWS_EMAIL": "bot@example.com",
            "EWS_PASSWORD": "secret",
            "WATCH_PATH": str(watch),
            "DEBOUNCE_SECONDS": "0.2",
            "LOG_LEVEL": "INFO",
        }
    )

    proc = subprocess.Popen(
        [str(venv_python), "-m", "reindex"],
        env=env,
        cwd=str(fake_root / "opt" / "it-consultant"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        time.sleep(0.4)
        (watch / "smoke.txt").write_text("ok", encoding="utf-8")
        time.sleep(0.8)
        assert proc.poll() is None, "reindex process exited unexpectedly"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

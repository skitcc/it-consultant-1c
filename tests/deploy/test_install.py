"""Installer layout tests against a fake root (never host systemd)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO_ROOT / "deploy" / "install.sh"


def _run_install(dest: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(INSTALL_SH), "--dest-dir", str(dest), *extra],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    return root


def test_layout_installs_gateway_sync_and_mail_units(fake_root: Path) -> None:
    result = _run_install(fake_root, "--layout-only")

    assert "layout-only" in result.stdout
    assert (fake_root / "var/lib/it-consultant").is_dir()
    env_text = (fake_root / "etc/it-consultant/.env").read_text(encoding="utf-8")
    assert "DOCUMENT_REGISTRY_PATH=/var/lib/it-consultant/registry.sqlite3" in env_text
    assert "WATCH_PATH=" not in env_text

    units = fake_root / "etc/systemd/system"
    assert (units / "api-gateway.service").is_file()
    assert (units / "knowledge-sync.service").is_file()
    assert (units / "mail-gateway.service").is_file()
    assert (units / "it-consultant.target").is_file()
    assert not (units / "reindex.service").exists()


def test_units_use_expected_python_modules(fake_root: Path) -> None:
    _run_install(fake_root, "--layout-only")
    units = fake_root / "etc/systemd/system"

    assert "python -m api_gateway" in (units / "api-gateway.service").read_text()
    assert "python -m knowledge_sync" in (units / "knowledge-sync.service").read_text()
    assert "python -m mail_gateway" in (units / "mail-gateway.service").read_text()
    target = (units / "it-consultant.target").read_text()
    assert "api-gateway.service knowledge-sync.service mail-gateway.service" in target
    assert "reindex" not in target


@pytest.mark.parametrize(
    ("service", "unit"),
    [
        ("api-gateway", "api-gateway.service"),
        ("knowledge-sync", "knowledge-sync.service"),
        ("mail-gateway", "mail-gateway.service"),
    ],
)
def test_only_selects_supported_service(
    fake_root: Path,
    service: str,
    unit: str,
) -> None:
    result = _run_install(
        fake_root,
        "--layout-only",
        "--only",
        service,
        "--enable",
    )
    assert f"would enable: {unit}" in result.stdout


def test_only_rejects_removed_reindex_service(fake_root: Path) -> None:
    result = subprocess.run(
        [
            str(INSTALL_SH),
            "--dest-dir",
            str(fake_root),
            "--layout-only",
            "--only",
            "reindex",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 2
    assert "unsupported --only service" in result.stderr


def test_layout_preserves_existing_env(fake_root: Path) -> None:
    env_path = fake_root / "etc/it-consultant/.env"
    env_path.parent.mkdir(parents=True)
    env_path.write_text("EWS_SERVER=keep-me\n", encoding="utf-8")

    _run_install(fake_root, "--layout-only")

    assert env_path.read_text(encoding="utf-8") == "EWS_SERVER=keep-me\n"


def test_undeploy_removes_installed_tree(fake_root: Path) -> None:
    _run_install(fake_root, "--layout-only")
    result = _run_install(fake_root, "--undeploy")

    assert "undeploy done" in result.stdout
    assert not (fake_root / "opt/it-consultant").exists()
    assert not (fake_root / "etc/it-consultant").exists()
    assert not (fake_root / "var/lib/it-consultant").exists()
    units = fake_root / "etc/systemd/system"
    assert not (units / "api-gateway.service").exists()
    assert not (units / "knowledge-sync.service").exists()
    assert not (units / "mail-gateway.service").exists()


def test_configure_updates_first_ews_values(fake_root: Path) -> None:
    result = subprocess.run(
        [
            str(INSTALL_SH),
            "--dest-dir",
            str(fake_root),
            "--layout-only",
            "--configure",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        input="\n".join(
            ["mail.example.com", "bot@example.com", r"DOMAIN\bot", "secret"]
        ),
    )

    assert "configure .env" in result.stdout
    env_text = (fake_root / "etc/it-consultant/.env").read_text(encoding="utf-8")
    assert "EWS_SERVER=mail.example.com" in env_text
    assert "EWS_EMAIL=bot@example.com" in env_text
    assert r"EWS_USERNAME=DOMAIN\bot" in env_text
    assert "EWS_PASSWORD=secret" in env_text

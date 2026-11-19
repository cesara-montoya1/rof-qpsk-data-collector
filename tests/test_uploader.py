"""Tests for GITA rsync-over-SSH uploader, Wi-Fi guard, and mandatory per-user SSH identity."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from common.uploader import (
    build_ssh_transport_args,
    check_gita_network_access,
    infer_remote_relative_path,
    resolve_identity_file,
    upload_dataset_to_gita,
    validate_dataset_upload_path,
)


def _make_dummy_key(tmp_path: Path, name: str = "id_ed25519_test") -> Path:
    key = tmp_path / name
    key.write_text("dummy-key-placeholder\n", encoding="utf-8")
    return key


def test_validate_dataset_upload_path_valid(tmp_path: Path):
    exp_dir = tmp_path / "rof_qpsk" / "uspl" / "exp1"
    exp_dir.mkdir(parents=True)
    assert validate_dataset_upload_path(exp_dir) == exp_dir.resolve()


def test_validate_dataset_upload_path_nonexistent(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        validate_dataset_upload_path(tmp_path / "missing")


def test_validate_dataset_upload_path_file_rejected(tmp_path: Path):
    f = tmp_path / "dummy.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_dataset_upload_path(f)


def test_resolve_identity_file_required_and_validated(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPTNET_GITA_KEY", raising=False)
    with pytest.raises(ValueError, match="identidad SSH"):
        resolve_identity_file(None)

    with pytest.raises(FileNotFoundError):
        resolve_identity_file(tmp_path / "nonexistent_key")

    key = _make_dummy_key(tmp_path)
    assert resolve_identity_file(key) == key.resolve()

    monkeypatch.setenv("OPTNET_GITA_KEY", str(key))
    assert resolve_identity_file(None) == key.resolve()


def test_infer_remote_relative_path():
    p1 = Path("/tmp/local_staging/rof_qpsk/uspl/nlna_step0p5db_650mhz_ch24")
    assert infer_remote_relative_path(p1) == "rof_qpsk/uspl/nlna_step0p5db_650mhz_ch24"
    assert (
        infer_remote_relative_path(Path("/tmp/my_exp"), tech_subdir="rof_16qam/dfb")
        == "rof_16qam/dfb/my_exp"
    )


def test_build_ssh_transport_args_with_identity():
    cmd = build_ssh_transport_args(identity_file="~/.ssh/id_ed25519_hanna")
    assert "-i" in cmd
    assert "IdentitiesOnly=yes" in cmd


@pytest.mark.parametrize("ssid", ["VisitanteUdeA", "SoyUdeA", "soyudea"])
def test_blocked_wifi_without_ethernet_aborts(ssid: str):
    with (
        patch("common.uploader.detect_active_wifi_ssid", return_value=ssid),
        patch("common.uploader.has_active_ethernet", return_value=False),
    ):
        can_proceed, warning = check_gita_network_access()
        assert can_proceed is False
        assert warning is not None
        assert "cable Ethernet" in warning
        assert "VisitanteUdeA" in warning and "SoyUdeA" in warning
        assert "22666" not in warning


def test_blocked_wifi_with_ethernet_warns_but_allows():
    with (
        patch("common.uploader.detect_active_wifi_ssid", return_value="SoyUdeA"),
        patch("common.uploader.has_active_ethernet", return_value=True),
    ):
        can_proceed, warning = check_gita_network_access()
        assert can_proceed is True
        assert warning is not None


def test_upload_dataset_to_gita_rejects_missing_identity(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPTNET_GITA_KEY", raising=False)
    exp_dir = tmp_path / "exp1"
    exp_dir.mkdir()
    ok, msg = upload_dataset_to_gita(exp_dir, dry_run=True)
    assert ok is False
    assert "identidad SSH" in msg


def test_upload_dataset_to_gita_dry_run(tmp_path: Path):
    exp_dir = tmp_path / "rof_qpsk" / "dfb" / "baseline_650mhz"
    exp_dir.mkdir(parents=True)
    key = _make_dummy_key(tmp_path, "id_hanna")

    with (
        patch("common.uploader.detect_active_wifi_ssid", return_value="HomeWiFi"),
        patch("common.uploader.resolve_ssh_user", return_value="hanna.colorado"),
    ):
        ok, msg = upload_dataset_to_gita(
            exp_dir,
            ssh_host="gita",
            ssh_user="hanna.colorado",
            identity_file=key,
            dry_run=True,
        )
        assert ok is True
        assert "DRY RUN" in msg
        assert "hanna.colorado@gita:/export2/camontoya/sofa_datasets/rof_qpsk/dfb/baseline_650mhz/" in msg
        assert "--chmod=Dg+s,ug+rwX,o-rwx" in msg
        assert str(key.resolve()) in msg


def test_upload_dataset_to_gita_aborts_on_blocked_wifi(tmp_path: Path):
    exp_dir = tmp_path / "exp1"
    exp_dir.mkdir()
    key = _make_dummy_key(tmp_path)
    with (
        patch("common.uploader.detect_active_wifi_ssid", return_value="VisitanteUdeA"),
        patch("common.uploader.has_active_ethernet", return_value=False),
    ):
        ok, msg = upload_dataset_to_gita(exp_dir, identity_file=key, dry_run=False)
        assert ok is False
        assert "VisitanteUdeA" in msg
        assert "cable Ethernet" in msg


@patch("common.uploader.subprocess.run")
def test_upload_dataset_to_gita_success_with_audit_log(mock_run: MagicMock, tmp_path: Path):
    exp_dir = tmp_path / "rof_qpsk" / "uspl" / "demo_exp"
    dist_dir = exp_dir / "0km"
    dist_dir.mkdir(parents=True)
    (dist_dir / "usrp_rx_gui_rof_qpsk_uspl_demo_exp_0km.npz").write_bytes(b"npz")
    (dist_dir / "sample.complex64").write_bytes(b"\x00" * 16)
    key = _make_dummy_key(tmp_path)

    mock_run.side_effect = [
        MagicMock(returncode=0, stdout="", stderr=""),  # mkdir -p
        MagicMock(returncode=0, stdout="ok", stderr=""),  # rsync
        MagicMock(returncode=0, stdout="", stderr=""),  # append upload_log.jsonl
    ]

    with (
        patch("common.uploader.detect_active_wifi_ssid", return_value=None),
        patch("common.uploader.shutil.which", return_value="/usr/bin/rsync"),
        patch("common.uploader.resolve_ssh_user", return_value="camontoya"),
    ):
        ok, msg = upload_dataset_to_gita(exp_dir, identity_file=key, dry_run=False)
        assert ok is True
        assert "camontoya" in msg
        assert "1 .npz" in msg
        assert "1 .complex64" in msg
        assert mock_run.call_count == 3
        rsync_call_args = mock_run.call_args_list[1][0][0]
        assert "--exclude=*.complex64" not in rsync_call_args

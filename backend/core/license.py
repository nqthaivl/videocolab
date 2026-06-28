import hashlib
import getpass
import glob
import logging
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile

from core import prefs
from core.config import DATA_DIR

logger = logging.getLogger("omnivoice.license")

SECRET_SALT = "video_clone_secret_salt_2026"
_MAC_RE = re.compile(r"^([0-9A-F]{2}:){5}[0-9A-F]{2}$")
_MACHINE_ID_RE = re.compile(r"^[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}$")
_FP_RE = re.compile(r"^[0-9A-F]{64}$")
_MACHINE_ID_PATH = os.path.join(DATA_DIR, "machine.id")

_session_token = os.environ.get("OMNIVOICE_SESSION_TOKEN", "")


def _normalize_mac(raw: str) -> str | None:
    mac = raw.strip().upper().replace("-", ":")
    if mac and mac != "00:00:00:00:00:00" and _MAC_RE.match(mac):
        return mac
    return None


def _collect_macs_getmac() -> list[str]:
    """Windows getmac — includes virtual adapters Node os.networkInterfaces() also sees."""
    macs: list[str] = []
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.check_output(
            ["getmac", "/fo", "csv", "/nh"],
            text=True,
            errors="replace",
            creationflags=flags,
        )
        for line in out.splitlines():
            if not line.strip():
                continue
            mac = _normalize_mac(line.split(",")[0].strip().strip('"'))
            if mac:
                macs.append(mac)
    except Exception:
        pass
    return macs


def _collect_macs() -> list[str]:
    """Collect MAC addresses — sorted unique set (matches Electron os.networkInterfaces())."""
    found: set[str] = set()
    try:
        import psutil

        for name in sorted(psutil.net_if_addrs().keys()):
            lowered = name.lower()
            if lowered in ("lo",) or lowered.startswith("loopback pseudo-interface"):
                continue
            for addr in psutil.net_if_addrs()[name]:
                if addr.family != psutil.AF_LINK:
                    continue
                mac = _normalize_mac(addr.address)
                if mac:
                    found.add(mac)
    except Exception:
        pass

    if sys.platform == "win32":
        found.update(_collect_macs_getmac())

    if not found and sys.platform != "win32":
        for addr_path in sorted(glob.glob("/sys/class/net/*/address")):
            name = addr_path.split("/")[-2]
            if name == "lo":
                continue
            try:
                with open(addr_path, encoding="utf-8") as f:
                    mac = _normalize_mac(f.read())
                if mac:
                    found.add(mac)
            except OSError:
                continue
    return sorted(found)


def _read_platform_machine_id() -> str | None:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
            ) as key:
                guid, _ = winreg.QueryValueEx(key, "MachineGuid")
            value = str(guid).strip().upper()
            return value or None
        except Exception:
            return None
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                text=True,
                errors="replace",
            )
            match = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', out)
            return match.group(1).strip().upper() if match else None
        except Exception:
            return None
    try:
        with open("/etc/machine-id", encoding="utf-8") as f:
            value = f.read().strip().upper()
        return value or None
    except OSError:
        return None


def get_hardware_fingerprint() -> str:
    """Live hardware fingerprint ΓÇö copied license files cannot satisfy this on another PC."""
    parts: list[str] = []
    platform_id = _read_platform_machine_id()
    if platform_id:
        parts.append(f"platform:{platform_id}")
    macs = _collect_macs()
    if macs:
        parts.append(f"mac:{'|'.join(macs)}")
    if not parts:
        parts.append(
            "fallback:"
            + socket.gethostname()
            + "|"
            + getpass.getuser()
            + "|"
            + sys.platform
            + "|"
            + platform.machine()
        )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest().upper()


def _read_stored_hardware_fingerprint() -> str | None:
    fp = prefs.get("license_hardware_fp")
    if not isinstance(fp, str):
        return None
    normalized = fp.strip().upper()
    return normalized if _FP_RE.match(normalized) else None


def _hardware_fingerprint_matches(stored: str | None) -> bool:
    if not stored:
        return True
    return stored == get_hardware_fingerprint()


def _read_persisted_machine_id() -> str | None:
    try:
        with open(_MACHINE_ID_PATH, encoding="utf-8") as f:
            lines = [line.strip() for line in f.read().splitlines() if line.strip()]
        if not lines:
            return None
        machine_id = lines[0].upper()
        if not _MACHINE_ID_RE.match(machine_id):
            return None
        if len(lines) >= 2:
            bound_fp = lines[1].upper()
            if _FP_RE.match(bound_fp) and bound_fp != get_hardware_fingerprint():
                logger.warning("machine.id belongs to another machine ΓÇö ignoring copied file")
                return None
        return machine_id
    except OSError:
        return None


def _persist_machine_id(machine_id: str) -> None:
    normalized = machine_id.strip().upper()
    if not _MACHINE_ID_RE.match(normalized):
        return
    os.makedirs(os.path.dirname(_MACHINE_ID_PATH) or DATA_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".machine-id.",
        suffix=".tmp",
        dir=os.path.dirname(_MACHINE_ID_PATH) or DATA_DIR,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{normalized}\n{get_hardware_fingerprint()}\n")
        os.replace(tmp, _MACHINE_ID_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _compute_machine_id_from_hardware() -> str:
    macs = _collect_macs()
    if macs:
        seed = "".join(macs)
    else:
        seed = socket.gethostname() + getpass.getuser() + sys.platform + platform.machine()
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest().upper()
    return f"{digest[:4]}-{digest[4:8]}-{digest[8:12]}"


def get_machine_id() -> str:
    """Stable Machine ID — persisted after activation; live hardware ID before."""
    saved = prefs.get("license_key")
    if isinstance(saved, str) and saved.strip():
        persisted = _read_persisted_machine_id()
        if persisted:
            return persisted
    return _compute_machine_id_from_hardware()


def _machine_id_for_activation() -> str:
    """Prefer Electron-synced machine.id (written before activate-internal)."""
    try:
        with open(_MACHINE_ID_PATH, encoding="utf-8") as f:
            synced = f.readline().strip().upper()
        if _MACHINE_ID_RE.match(synced):
            return synced
    except OSError:
        pass
    return _compute_machine_id_from_hardware()

def verify_license_key(key: str, *, for_activation: bool = False) -> bool:
    if not key:
        return False
    if not for_activation:
        stored_fp = _read_stored_hardware_fingerprint()
        if not _hardware_fingerprint_matches(stored_fp):
            logger.warning("License hardware fingerprint mismatch ΓÇö activation rejected")
            return False
    normalized_key = key.replace("-", "").replace(" ", "").upper()
    if len(normalized_key) != 16:
        return False
    machine_id = _machine_id_for_activation() if for_activation else get_machine_id()
    normalized_machine = machine_id.replace("-", "").replace(" ", "").upper()
    hash_input = (normalized_machine + SECRET_SALT).encode("utf-8")
    expected = hashlib.sha256(hash_input).hexdigest().upper()[:16]
    if normalized_key == expected:
        _persist_machine_id(machine_id)
        if not _read_stored_hardware_fingerprint():
            prefs.set_("license_hardware_fp", get_hardware_fingerprint())
        return True
    return False


def _persist_activation_record(key: str) -> None:
    prefs.set_("license_key", key)
    prefs.set_("license_hardware_fp", get_hardware_fingerprint())
    _persist_machine_id(get_machine_id())


def _bootstrap_activation() -> bool:
    """True when env says activated OR a valid key is persisted in prefs.json."""
    if os.environ.get("OMNIVOICE_ACTIVATED") == "1":
        return True
    saved = prefs.get("license_key")
    if saved and verify_license_key(str(saved)):
        logger.info("License restored from prefs.json on startup")
        return True
    return False


# Standalone (no session token): always activated for dev/Colab notebook.
# Electron mode: env var + prefs fallback.
if not _session_token:
    _is_activated = True
else:
    _is_activated = _bootstrap_activation()


def verify_session_token(token: str) -> bool:
    if not _session_token:
        return False
    return token == _session_token


def is_activated() -> bool:
    """Kiểm tra xem phần mềm đã được kích hoạt thành công hay chưa."""
    return _is_activated


def activate_software(key: str) -> bool:
    """Thử kích hoạt phần mềm bằng một khóa mới (chỉ dùng trong chế độ standalone dev)."""
    global _is_activated
    if not verify_license_key(key):
        return False
    _persist_activation_record(key)
    _is_activated = True
    logger.info("Software activated in standalone mode")
    return True


def activate_software_internal(key: str) -> bool:
    """Kích hoạt phần mềm nội bộ sau khi JS đã xác thực thành công."""
    global _is_activated
    if not verify_license_key(key, for_activation=True):
        logger.warning("activate_software_internal: key failed Python verification")
        return False
    _persist_activation_record(key)
    _is_activated = True
    logger.info("Software activated successfully via JS verification")
    return True

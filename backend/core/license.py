import logging
import os
from core import prefs

logger = logging.getLogger("omnivoice.license")

# Initially load state from environment variables set by Electron
_session_token = os.environ.get("OMNIVOICE_SESSION_TOKEN", "")

# In standalone mode (no session token), default to activated for dev/testing ease.
# In Electron mode, require OMNIVOICE_ACTIVATED == "1"
if not _session_token:
    _is_activated = True
else:
    _is_activated = os.environ.get("OMNIVOICE_ACTIVATED") == "1"

def verify_session_token(token: str) -> bool:
    if not _session_token:
        return False
    return token == _session_token

def verify_license_key(key: str) -> bool:
    # Deprecated in Python, verification now happens in JS.
    # Return True to avoid breaking callers, but activation state is managed via _is_activated.
    return _is_activated

def is_activated() -> bool:
    """Kiểm tra xem phần mềm đã được kích hoạt thành công hay chưa."""
    return _is_activated

def activate_software(key: str) -> bool:
    """Thử kích hoạt phần mềm bằng một khóa mới (chỉ dùng trong chế độ standalone dev)."""
    global _is_activated
    prefs.set_("license_key", key)
    _is_activated = True
    logger.info("Software activated in standalone mode with key: %s", key)
    return True

def activate_software_internal(key: str) -> bool:
    """Kích hoạt phần mềm nội bộ sau khi JS đã xác thực thành công."""
    global _is_activated
    prefs.set_("license_key", key)
    _is_activated = True
    logger.info("Software activated successfully via JS verification with key: %s", key)
    return True

from __future__ import annotations
import json
import time
import urllib.request

from PyQt6.QtCore import QThread, pyqtSignal, QSettings

_GITHUB_API = "https://api.github.com/repos/brentmantooth/AstroImageLab/releases/latest"
_CHECK_INTERVAL_S = 7 * 86400  # 7 days


def _vtuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.lstrip("v").split(".") if x.isdigit())


class UpdateCheckerThread(QThread):
    update_available = pyqtSignal(str, str)  # (current_version, latest_version)

    def __init__(self, current_version: str, parent=None):
        super().__init__(parent)
        self._current = current_version

    def run(self) -> None:
        settings = QSettings("FilterImageComparator", "FilterImageComparator")
        last = settings.value("last_version_check", 0.0, type=float)
        now = time.time()
        if now - last < _CHECK_INTERVAL_S:
            return

        try:
            req = urllib.request.Request(
                _GITHUB_API,
                headers={"User-Agent": "AstroImageLab-UpdateChecker"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            latest = data.get("tag_name", "").lstrip("v")
            if not latest:
                return
            settings.setValue("last_version_check", now)
            if _vtuple(latest) > _vtuple(self._current):
                self.update_available.emit(self._current, latest)
        except Exception:
            pass  # network unavailable or rate-limited — fail silently

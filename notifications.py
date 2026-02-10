import structlog
import sys
from typing import Optional

# Windows notifications
try:
    from winotify import Notification, audio
    HAS_WINOTIFY = True
except ImportError:
    HAS_WINOTIFY = False

try:
    from win10toast_py3 import ToastNotifier
    HAS_TOAST = True
except ImportError:
    HAS_TOAST = False

# macOS notifications
try:
    import pync
    HAS_PYNC = True
except ImportError:
    HAS_PYNC = False

# Linux notifications
try:
    import notify2
    HAS_NOTIFY2 = True
except ImportError:
    HAS_NOTIFY2 = False


class DesktopNotifier:
    """Cross-platform desktop notifications."""

    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.toaster = None

        # Initialize based on platform
        if sys.platform == "win32":
            if HAS_WINOTIFY:
                self.platform = "windows_winotify"
            elif HAS_TOAST:
                self.toaster = ToastNotifier()
                self.platform = "windows_toast"
            else:
                self.platform = "none"
        elif sys.platform == "darwin" and HAS_PYNC:
            self.platform = "macos"
        elif HAS_NOTIFY2:
            try:
                notify2.init("Fortuna")
                self.platform = "linux"
            except Exception:
                self.platform = "none"
        else:
            self.platform = "none"
            self.logger.warning("No native notification system available or supported", platform=sys.platform)

    def send(self, alert: Optional[dict] = None, **kwargs):
        """
        Send a desktop notification.
        Supports both a dictionary 'alert' or direct keyword arguments.
        """
        if alert:
            kwargs.update(alert)

        title = kwargs.get("title", "Fortuna Alert")
        message = kwargs.get("message", kwargs.get("msg", ""))
        urgency = kwargs.get("urgency", "normal")

        # Always log the notification
        self.logger.info("NOTIFICATION", title=title, message=message, urgency=urgency)

        try:
            if self.platform.startswith("windows"):
                self._send_windows(title, message, urgency)
            elif self.platform == "macos":
                self._send_macos(title, message, urgency)
            elif self.platform == "linux":
                self._send_linux(title, message, urgency)
        except Exception as e:
            self.logger.error("Notification delivery failed", error=str(e))

    def _send_windows(self, title: str, message: str, urgency: str):
        """Windows toast notification."""
        if self.platform == "windows_winotify":
            toast = Notification(
                app_id="Fortuna Intelligence",
                title=title,
                msg=message,
                duration="long" if urgency == "high" else "short"
            )
            if urgency == "high":
                toast.set_audio(audio.Reminder, loop=False)
            toast.show()
        elif self.platform == "windows_toast":
            duration = 10 if urgency == "high" else 5
            self.toaster.show_toast(
                title=title,
                msg=message,
                duration=duration,
                threaded=True
            )

    def _send_macos(self, title: str, message: str, urgency: str):
        """macOS notification center."""
        pync.notify(
            message,
            title=title,
            sound="default" if urgency == "high" else None
        )

    def _send_linux(self, title: str, message: str, urgency: str):
        """Linux notification daemon."""
        n = notify2.Notification(title, message)
        if urgency == "high":
            n.set_urgency(notify2.URGENCY_CRITICAL)
        elif urgency == "low":
            n.set_urgency(notify2.URGENCY_LOW)
        n.show()

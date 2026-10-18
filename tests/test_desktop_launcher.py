import threading
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import desktop_launcher


class DesktopLauncherTests(unittest.TestCase):
    def test_server_readiness_succeeds_if_launch_thread_exits_after_http_ready(self):
        thread = threading.Thread(target=lambda: None)
        thread.start()
        thread.join()
        response = MagicMock()
        response.__enter__.return_value.status = 200

        with patch.object(
            desktop_launcher.urllib.request, "urlopen", return_value=response
        ):
            desktop_launcher.wait_for_server("http://127.0.0.1/", thread, [])

    def test_server_readiness_fails_if_thread_stops_before_http_ready(self):
        thread = threading.Thread(target=lambda: None)
        thread.start()
        thread.join()

        with (
            patch.object(
                desktop_launcher.urllib.request, "urlopen",
                side_effect=URLError("connection refused"),
            ),
            self.assertRaisesRegex(RuntimeError, "service stopped during startup"),
        ):
            desktop_launcher.wait_for_server("http://127.0.0.1/", thread, [])


if __name__ == "__main__":
    unittest.main()

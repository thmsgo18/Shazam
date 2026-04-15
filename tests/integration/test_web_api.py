from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from webapp.backend import server


class WebApiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(server.app)

    def test_health_endpoint_returns_ok(self) -> None:
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_config_endpoint_proxies_ui_config(self) -> None:
        with patch("webapp.backend.server.get_ui_config", return_value={"listen_duration": 15}):
            response = self.client.get("/api/config")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"listen_duration": 15})

    def test_identify_endpoint_returns_payload(self) -> None:
        payload = {"results": [{"track_id": "track_1"}], "confident": True, "recommendations": []}

        with patch("webapp.backend.server._to_wav", return_value=None), \
             patch("webapp.backend.server.build_identification_response", return_value=payload):
            response = self.client.post(
                "/api/identify",
                files={"file": ("query.wav", b"fake-audio-bytes", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)


if __name__ == "__main__":
    unittest.main()

"""POST /stt with the upstream mocked. Run with `python -m unittest`."""

import io
import sys
import types
import unittest
from unittest import mock

import requests

if "app.tts_local" not in sys.modules:
    stub = types.ModuleType("app.tts_local")
    stub.load_models = lambda: None
    stub.generate_audio = lambda text, path: False
    stub.model = None
    stub.tokenizer = None
    sys.modules["app.tts_local"] = stub

from app import config, stt_api  # noqa: E402
from app import main as app_main  # noqa: E402

SECRET_WORDS = "silav tu çawa yî"


def _upstream(status=200, body=None):
    response = mock.Mock(ok=status < 400, status_code=status, text="boom")
    response.json.return_value = body if body is not None else {"text": SECRET_WORDS, "detected_dialect": "kurmanji"}
    return response


class SttRouteTest(unittest.TestCase):
    def setUp(self):
        self.client = app_main.app.test_client()
        patcher = mock.patch.object(config, "KURDISH_TTS_STT_KEY", "test-key")
        patcher.start()
        self.addCleanup(patcher.stop)

    def post(self, course="kmr", data=b"RIFFxxxx"):
        return self.client.post(
            "/stt",
            data={"file": (io.BytesIO(data), "clip.wav"), "course": course},
            content_type="multipart/form-data",
        )

    def test_happy_path_one_shape_and_forwarded_fields(self):
        with mock.patch.object(stt_api.requests, "post", return_value=_upstream()) as post:
            response = self.post()
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["text"], SECRET_WORDS)
        self.assertEqual((body["engine"], body["course"], body["truncated"]), ("kurdishtts", "kmr", False))
        self.assertEqual(body["raw"]["detected_dialect"], "kurmanji")
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["data"], {"dialect": "kurmanji"})
        self.assertEqual(kwargs["headers"], {"x-api-key": "test-key"})

    def test_503_without_the_key(self):
        with mock.patch.object(config, "KURDISH_TTS_STT_KEY", ""):
            with mock.patch.object(stt_api.requests, "post") as post:
                response = self.post()
        self.assertEqual(response.status_code, 503)
        post.assert_not_called()

    def test_501_for_heb(self):
        response = self.post(course="heb")
        self.assertEqual(response.status_code, 501)
        self.assertEqual(response.get_json(), {"error": "no engine for heb yet"})

    def test_400_for_an_unknown_course_or_no_file(self):
        self.assertEqual(self.post(course="xx").status_code, 400)
        response = self.client.post("/stt", data={"course": "kmr"}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_413_over_ten_megabytes(self):
        with mock.patch.object(stt_api.requests, "post") as post:
            response = self.post(data=b"0" * (stt_api.MAX_CLIP_BYTES + 1))
        self.assertEqual(response.status_code, 413)
        post.assert_not_called()

    def test_504_on_timeout(self):
        with mock.patch.object(stt_api.requests, "post", side_effect=requests.exceptions.Timeout):
            self.assertEqual(self.post().status_code, 504)

    def test_upstream_error_passes_status_and_body(self):
        with mock.patch.object(stt_api.requests, "post", return_value=_upstream(401, {"error": "bad key"})):
            response = self.post()
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.get_json(), {"error": "bad key"})

    def test_no_transcript_in_the_log(self):
        with mock.patch.object(stt_api.requests, "post", return_value=_upstream()):
            with self.assertLogs(level="INFO") as logs:
                self.post()
        self.assertTrue(logs.output)
        self.assertNotIn("silav", "\n".join(logs.output))

    def test_health_reports_the_key(self):
        self.assertEqual(self.client.get("/health").get_json()["stt"], {"kmr": True})
        with mock.patch.object(config, "KURDISH_TTS_STT_KEY", ""):
            self.assertEqual(self.client.get("/health").get_json()["stt"], {"kmr": False})

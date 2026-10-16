"""Speech to text behind one door: `transcribe(course, ...)` picks the engine.

One function per engine, one table by course. A new engine (Google Cloud Speech
for Hebrew, behind GOOGLE_CLOUD_TTS_STT_KEY) is one function and one table row;
the route never changes. Nothing here logs or keeps a transcript.
"""

import time
import requests
from . import config
from .config import log

STT_URL = "https://www.kurdishtts.com/api/stt-proxy"
STT_TIMEOUT = 90
MAX_CLIP_BYTES = 10 * 1024 * 1024
COURSES = ("kmr", "heb")


class SttError(Exception):
    """An answer the route passes on: a status and a JSON body."""

    def __init__(self, status: int, body: dict):
        super().__init__(status)
        self.status = status
        self.body = body


def _kurdishtts(filename: str, data: bytes, content_type: str) -> dict:
    if not config.KURDISH_TTS_STT_KEY:
        raise SttError(503, {"error": "KURDISH_TTS_STT_KEY is not set on the server"})
    try:
        response = requests.post(
            STT_URL,
            files={"file": (filename, data, content_type)},
            data={"dialect": "kurmanji"},
            headers={"x-api-key": config.KURDISH_TTS_STT_KEY},
            timeout=STT_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        raise SttError(504, {"error": "the speech engine timed out"})
    except requests.exceptions.RequestException:
        raise SttError(502, {"error": "the speech engine could not be reached"})
    if not response.ok:
        try:
            body = response.json()
        except ValueError:
            body = {"error": response.text[:200]}
        raise SttError(response.status_code, body)
    raw = response.json()
    return {
        "text": raw.get("text", ""),
        "engine": "kurdishtts",
        "course": "kmr",
        "truncated": bool(raw.get("truncated", False)),
        "raw": raw,
    }


def _no_engine(filename: str, data: bytes, content_type: str) -> dict:
    raise SttError(501, {"error": "no engine for heb yet"})


ENGINES = {"kmr": _kurdishtts, "heb": _no_engine}


def transcribe(course: str, filename: str, data: bytes, content_type: str) -> dict:
    if len(data) > MAX_CLIP_BYTES:
        raise SttError(413, {"error": "clip is over 10 MB"})
    start = time.perf_counter()
    try:
        answer = ENGINES[course](filename, data, content_type)
    except SttError as error:
        log(f"stt {course}: {error.status} | {len(data)} bytes | {time.perf_counter() - start:.2f}s")
        raise
    log(
        f"stt {course}: ok | {len(data)} bytes | {len(answer['text'])} chars | "
        f"{time.perf_counter() - start:.2f}s"
    )
    return answer

"""The voices this service speaks: the one table, the one place.

An allow-list, not a pass-through: a request for any other voice is refused,
so nobody can fill the cache with arbitrary voices. The default voice keeps
every key it had before voices existed (clip, manifest, response cache), so
nothing already cached is rebuilt or renamed.

The four studio voices were picked for the app's scene speakers (kmj-daily
backlog P50, 2026-10-02); they exist only on the engine's v5.
"""

from typing import NamedTuple

DIALECT = "kurmanji"
DEFAULT_VOICE = "kurmanji_236"


class Voice(NamedTuple):
    model_version: str
    # Whether the authenticated endpoint accepts this voice on our key, so a
    # rate-limited free request may fall through to it and a sentence over the
    # free limit may go there. Checked by one real request per voice
    # (2026-10-02): v5 answered 403 "TTS v5 requires a paid API plan".
    paid: bool


VOICES = {
    DEFAULT_VOICE: Voice("v4", paid=True),
    "studio_elder_m": Voice("v5", paid=False),
    "studio_docu_m": Voice("v5", paid=False),
    "studio_host_f": Voice("v5", paid=False),
    "studio_teacher_f": Voice("v5", paid=False),
}


class UnknownVoice(ValueError):
    """A voice id that is not in the table."""


def resolve(voice: str | None) -> str:
    """The voice id a request asked for; no voice (or an empty one) is the default."""
    voice = (voice or "").strip()
    if not voice:
        return DEFAULT_VOICE
    if voice not in VOICES:
        raise UnknownVoice(voice)
    return voice


def model_version(voice: str) -> str:
    return VOICES[voice].model_version


def accepts_paid(voice: str) -> bool:
    return VOICES[voice].paid


def variant(voice: str) -> str:
    """What the response cache keys on beside the text: a new voice is a new request."""
    return f"{DIALECT}/{voice}"


def file_tag(voice: str) -> str:
    """The part of a clip or manifest name that carries the voice.

    Empty for the default voice, so its names stay exactly as they were.
    """
    return "" if voice == DEFAULT_VOICE else f".{voice}"


def allowed() -> list:
    return list(VOICES)

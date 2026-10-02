"""Whether a clip's word times can be trusted, judged against its own sound.

The app lights each word as it is spoken from the manifest's `words`. A word
lit at the wrong moment is worse than no lighting at all (his word,
2026-10-02), so a chunk whose engine word times disagree with the clip's
sound gets `words: null` (the app then lights the whole line) and keeps the
engine's words under `rejected_words`, with the reasons, for later study.

Pure: everything here works on the clip's own timeline, from a chunk's
mapped words and the clip's sound islands (`tts_api._sound_islands`), and
never decodes, reads or writes a file. Thresholds were set 2026-10-02 on 79
cached clips (34 default voice, 45 across ten v5 voices), decoded and
islanded the same way, with word times from their manifests.
"""

# Bump when a rule below changes: every manifest whose `check.version` is
# older is checked again (lazily, and by the startup sweep).
VERSION = 1

# A word shorter than this cannot be lit. The app uses the same floor; the
# shortest real word measured was 40ms ("Tu", studio_docu_m), while the two
# known bad lines carry words of exactly 0ms ("ku", "me").
MIN_WORD_MS = 30

# An island of sound that starts this long after the last word's end is
# speech the words do not cover. Measured: a real final stop's release stands
# as its own island within about 110ms of the word end; the widest gap in a
# good clip was 250ms (a 5ms tick, kurmanji_236, "Ez kurmancî dixwînim.");
# the stray sound in the bad line ("Tu li ku dijî?", studio_docu_m) starts
# 395ms after its last word.
SOUND_AFTER_WORDS_MS = 300

# Islands this short are the engine's click (10 to 25ms, measured 2026-09-18
# and 2026-10-02), not speech: they cannot hide a word, so they never count
# as sound after the words.
TICK_MAX_MS = 25

# Words that end this long before their chunk's kept audio does leave sound
# the lighting never reaches. The app's own bound is 600ms before `clip_ms`;
# the last chunk ends LEAD_OUT_MS (80ms) before the clip, so 520ms here is
# the same bound per chunk. Measured gap from the last word end to the
# chunk's end: 0 to 300ms in good clips (median about 110ms), 920ms in the
# bad line.
WORDS_END_BEFORE_CHUNK_END_MS = 520

# The reasons that mean the clip carries stray sound after its words: for a
# voice other than the default, such a clip is synthesised once more.
STRAY_TAIL = ("sound_after_words", "words_end_early")


def chunk_span(chunk: dict) -> tuple:
    """Where a manifest chunk's kept audio starts and ends on the clip."""
    pieces = chunk["pieces"]
    last = pieces[-1]
    return pieces[0]["clip_start_ms"], last["clip_start_ms"] + last["raw_end_ms"] - last["raw_start_ms"]


def _reason(code: str, word: dict | None = None, **extra) -> dict:
    reason = {"code": code}
    if word is not None:
        reason.update(word=word["word"], start_ms=word["start_ms"], end_ms=word["end_ms"])
    reason.update(extra)
    return reason


def _has_sound(word: dict, islands: list) -> bool:
    return any(start < word["end_ms"] and end > word["start_ms"] for start, end in islands)


def reasons(words: list, span: tuple, islands: list) -> list:
    """Why a chunk's word times cannot be trusted; empty when they can.

    `words`: the chunk's mapped words (`start_ms`, `end_ms`, clip timeline).
    `span`: the chunk's kept audio on the clip (`chunk_span`).
    `islands`: (start_ms, end_ms) sound islands of the whole clip.
    """
    if not words:
        return []
    found = []
    inside = [(start, end) for start, end in islands if end > span[0] and start < span[1]]
    for word in words:
        if word["end_ms"] - word["start_ms"] < MIN_WORD_MS:
            found.append(_reason("short_word", word))
        elif not _has_sound(word, inside):
            found.append(_reason("silent_word", word))
    last_end = max(word["end_ms"] for word in words)
    for start, end in inside:
        if start - last_end > SOUND_AFTER_WORDS_MS and end - start > TICK_MAX_MS:
            found.append(_reason("sound_after_words", island_start_ms=start, island_end_ms=end))
    if span[1] - last_end > WORDS_END_BEFORE_CHUNK_END_MS:
        found.append(_reason("words_end_early", last_word_end_ms=last_end, chunk_end_ms=span[1]))
    return found


def _engine_words(chunk: dict) -> list | None:
    """The engine's words, whether trusted (`words`) or set aside before."""
    if chunk.get("rejected_words") is not None:
        return chunk["rejected_words"]
    return chunk.get("words")


def apply(manifest: dict, islands: list, attempts: list | None = None, regenerated: bool = False) -> dict:
    """Check every chunk, withhold the words of each that fails, and record it.

    Idempotent: the engine's words are read back from `rejected_words` when
    an earlier check set them aside, so a chunk that passes now gets them back.
    """
    all_reasons = []
    for chunk in manifest["chunks"]:
        words = _engine_words(chunk)
        found = reasons(words, chunk_span(chunk), islands) if words else []
        chunk.pop("rejected_words", None)
        chunk.pop("rejected_reasons", None)
        chunk["words"] = words if not found else None
        if found:
            chunk["rejected_words"] = words
            chunk["rejected_reasons"] = found
        all_reasons.extend({"chunk": chunk["index"], **reason} for reason in found)
    manifest["check"] = {
        "version": VERSION,
        "ok": not all_reasons,
        "reasons": all_reasons,
        "attempts": attempts or manifest.get("check", {}).get("attempts") or [1] * len(manifest["chunks"]),
        "regenerated": regenerated or bool(manifest.get("check", {}).get("regenerated")),
    }
    return manifest


def is_current(manifest: dict) -> bool:
    return manifest.get("check", {}).get("version") == VERSION


def has_stray_tail(manifest: dict) -> bool:
    return any(reason["code"] in STRAY_TAIL for reason in manifest.get("check", {}).get("reasons", []))

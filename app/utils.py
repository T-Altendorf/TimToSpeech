import re
import hashlib
from pathlib import Path
from .config import CACHE_DIR

# Numbers to words mapping
num2word = {
    "0": "sifir",
    "1": "yek",
    "2": "du",
    "3": "sê",
    "4": "çar",
    "5": "pênc",
    "6": "şeş",
    "7": "heft",
    "8": "heşt",
    "9": "neh",
    "10": "deh",
}


def replace_numbers_with_words(text):
    """Replace numbers with their word equivalents"""

    def repl(match):
        num = match.group()
        return num2word.get(num, num)

    return re.sub(r"\b\d+\b", repl, text)


# Abbreviations
abbrev_as_word = {
    "KCK": "Keceke",
    "PKK": "Pekeke",
    "PAJK": "Pajek",
    "PYD": "Peyede",
    "YPG": "Yepege",
    "YPJ": "Yepeje",
    "HDP": "Hedepe",
    "DBP": "Debepe",
    "KDP": "Kedepe",
    "PDK": "Pedeke",
    "PUK": "Pûk",
    "YNK": "Yeneke",
    "TAK": "Tak",
    "PJAK": "Pejak",
    "ENKS": "Enekese",
    "TEV-DEM": "Tevdem",
    "KOMKAR": "Komkar",
    "NATO": "Nato",
    "UNESCO": "Yunesko",
    "UNICEF": "Yunîsef",
    "VOA": "Voa",
    "RAM": "Rem",
    "ram": "Rem",
}

abbrev_spelled = {
    "UN": "Û En",
    "EU": "E Û",
    "NGO": "En Cî O",
    "KRG": "Ke Re Ge",
    "BBC": "Bî Bî Sî",
    "CNN": "Sî En En",
    "DW": "De We",
    "TRT": "Te Re Te",
    "RT": "Er Te",
    "USB": "U Se Be",
    "PDF": "Pe De Fe",
    "AI": "A Î",
    "IT": "Ay Tî",
    "HTTP": "He Te Te Pe",
    "HTML": "He Te Me Le",
    "URL": "U Re Le",
    "IP": "Ay Pî",
    "CPU": "Sî Pî U",
    "GPU": "Cî Pî U",
    "SMS": "Es Em Es",
    "GPS": "Cî Pî Es",
}

abbrev_map = {}
abbrev_map.update(abbrev_as_word)
abbrev_map.update(abbrev_spelled)


def expand_abbreviations(text: str) -> str:
    """Expand abbreviations to their full forms"""
    for abbr, full in abbrev_map.items():
        pattern = r"(?<!\w)" + re.escape(abbr) + r"(?!\w)"
        text = re.sub(pattern, full, text)
    return text


def normalize_text(text: str) -> str:
    """Normalize quotation marks and apostrophes"""
    text = text.replace(""", "\"").replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")
    return text


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline"""
    text = normalize_text(text)
    text = replace_numbers_with_words(text)
    text = expand_abbreviations(text)
    return text


# Bump when the audio pipeline changes what a clip sounds like. The tag is part
# of the file name, so every clip made by an older pipeline stops being served
# and is rebuilt on the next request. 2: the engine's head artifact is trimmed
# and joins happen at true silence (2026-09-18). 3: the same burst is also cut
# from between sentences inside one clip (2026-09-19). 4: a burst cut fades
# instead of splicing raw silence, which was not true zero and could still
# step (2026-09-19). Older files are deleted when the service starts
# (`prune_stale_cache`), so a deploy cleans up after itself.
AUDIO_VERSION = 4


def get_cache_path(text: str) -> Path:
    """Generate cache file path based on text hash and the audio version"""
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    return CACHE_DIR / f"{text_hash}{_cache_suffix()}"


def _cache_suffix() -> str:
    return f".a{AUDIO_VERSION}.mp3"


def prune_stale_cache() -> int:
    """
    Delete every cached clip that is not of the current audio version.

    Runs once when the service starts. A deploy that bumps AUDIO_VERSION thus
    empties the old cache by itself; every text is generated again on its next
    request. Several Gunicorn workers may run this at once, so a file that is
    already gone is fine.
    """
    removed = 0
    for path in CACHE_DIR.glob("*.mp3"):
        if path.name.endswith(_cache_suffix()):
            continue
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            pass
    return removed


def cache_counts() -> dict:
    """How many cached clips are of the current audio version, and how many are not."""
    names = [path.name for path in CACHE_DIR.glob("*.mp3")]
    current = sum(1 for name in names if name.endswith(_cache_suffix()))
    return {"current": current, "stale": len(names) - current}

"""`POST /warm/bulk` and `GET /warm/bulk`: the night lane's list (`app/night.py`).

Both need `Authorization: Bearer <WARM_BULK_KEY>`; with no key set the
endpoint answers 503, a wrong or missing key 401. Answers are counts only.
"""
import hmac

from flask import Blueprint, jsonify, request

from . import config, night
from .night_list import APPEND, REPLACE

MAX_BULK_TEXTS = 50_000
MAX_BULK_BYTES = 2 * 1024 * 1024

bulk = Blueprint("bulk", __name__)


def _refused():
    """None when the request carries the key; else the refusal to send."""
    key = config.WARM_BULK_KEY
    if not key:
        return jsonify({"error": "WARM_BULK_KEY is not set on this server"}), 503
    header = request.headers.get("Authorization", "")
    given = header[len("Bearer "):] if header.startswith("Bearer ") else ""
    if not hmac.compare_digest(given.encode(), key.encode()):
        return jsonify({"error": "Unauthorized"}), 401
    return None


def _body() -> tuple:
    """(texts, mode, None) or (None, None, the refusal)."""
    if (request.content_length or 0) > MAX_BULK_BYTES:
        return None, None, (jsonify({"error": f"Body over {MAX_BULK_BYTES} bytes"}), 413)
    body = request.get_json(silent=True)
    texts = body.get("texts") if isinstance(body, dict) else None
    mode = body.get("mode", APPEND) if isinstance(body, dict) else None
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        return None, None, (jsonify({"error": "'texts' must be a list of strings"}), 400)
    if len(texts) > MAX_BULK_TEXTS:
        return None, None, (jsonify({"error": f"At most {MAX_BULK_TEXTS} texts"}), 413)
    if mode not in (APPEND, REPLACE):
        return None, None, (jsonify({"error": "'mode' must be append or replace"}), 400)
    return [text.strip() for text in texts], mode, None


@bulk.route("/warm/bulk", methods=["POST"])
def add_texts():
    """Lists texts for the night lane, in the default voice."""
    refused = _refused()
    if refused:
        return refused
    texts, mode, refused = _body()
    if refused:
        return refused
    counts = night.LANE.items.add(texts, mode, night.is_cached)
    night.LANE.start()
    return jsonify(counts)


@bulk.route("/warm/bulk", methods=["GET"])
def progress():
    refused = _refused()
    if refused:
        return refused
    return jsonify(night.LANE.progress())

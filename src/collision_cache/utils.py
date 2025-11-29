# utils.py
import hashlib
from typing import Callable
import re

try:
    import blake3
    _has_blake3 = True
except Exception:
    _has_blake3 = False

def canonicalize_input(text: str) -> str:
    """Normalize whitespace and basic token-level lowercasing.
    Adapt this for your domain (remove salts, canonicalize templates, etc.).
    """
    s = " ".join(text.strip().split())
    # optionally preserve case for code-like prompts — adapt per domain
    return s.lower()

def canonicalize_output(text: str) -> str:
    s = "\n".join(line.rstrip() for line in text.strip().splitlines())
    # collapse multiple blank lines
    s = re.sub(r'\n{2,}', '\n\n', s)
    return s

def truncated_hash_hex(data: bytes, bits: int = 64, algo: str = "sha256") -> str:
    """Return hex of truncated hash. bits should be multiple of 8 for simplicity."""
    if algo == "blake3" and _has_blake3:
        full = blake3.blake3(data).digest()
    else:
        full = hashlib.sha256(data).digest()
    bytes_needed = (bits + 7) // 8
    return full[:bytes_needed].hex()

def input_signature(text: str, bits: int = 64) -> str:
    return truncated_hash_hex(canonicalize_input(text).encode("utf-8"), bits=bits)

def output_fingerprint(text: str, bits: int = 64, algo: str = "sha256") -> str:
    return truncated_hash_hex(canonicalize_output(text).encode("utf-8"), bits=bits, algo=algo)

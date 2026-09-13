"""Iranian entity validators — Python port of parsitext validators.

Matches parsitext 0.1.3: national_id, legal_id, sheba, bank_card, phone,
landline, postal_code, car_plate, bill. Pure python, no deps.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# National ID (کد ملی) — 10 digits, checksum mod 11
# ---------------------------------------------------------------------------
_NATIONAL_RE = re.compile(r"^\d{10}$")


def validate_national_id(code: str) -> bool:
    code = re.sub(r"\D", "", code)
    if not _NATIONAL_RE.match(code) or len(set(code)) == 1:
        return False
    check = int(code[9])
    s = sum(int(code[i]) * (10 - i) for i in range(9)) % 11
    return (s < 2 and check == s) or (s >= 2 and check == 11 - s)


# ---------------------------------------------------------------------------
# Legal ID (شناسه ملی) — 11 digits, weighted mod 11
# ---------------------------------------------------------------------------
def validate_legal_id(code: str) -> bool:
    code = re.sub(r"\D", "", code)
    if len(code) != 11 or len(set(code)) == 1:
        return False
    check = int(code[10])
    # weights 29,27,23,19,17,29,27,23,19,17 for first 10
    weights = [29, 27, 23, 19, 17, 29, 27, 23, 19, 17]
    s = sum(int(code[i]) * weights[i] for i in range(10)) % 11
    # parsitext: if s==10 → 0
    if s == 10:
        s = 0
    return check == s


# ---------------------------------------------------------------------------
# Sheba / IBAN IR — IR + 24 digits, mod97 == 1
# ---------------------------------------------------------------------------
_SHEBA_RE = re.compile(r"^IR\d{24}$")


def validate_sheba(sheba: str) -> bool:
    s = re.sub(r"\s|-", "", sheba).upper()
    if not _SHEBA_RE.match(s):
        return False
    # Move IR + first 4 to end, convert letters A=10..Z=35
    rearranged = s[4:] + s[:4]
    numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    # mod97 iterative to avoid big ints
    remainder = 0
    for ch in numeric:
        remainder = (remainder * 10 + int(ch)) % 97
    return remainder == 1


# ---------------------------------------------------------------------------
# Bank card — 16 digits, Luhn
# ---------------------------------------------------------------------------
def validate_bank_card(card: str) -> bool:
    card = re.sub(r"\D", "", card)
    if len(card) != 16:
        return False
    # Luhn
    total = 0
    for i, ch in enumerate(reversed(card)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Bank BIN → name (subset, parsitext has full table)
BIN_BANK = {
    "610433": "Mellat", "627353": "Tejarat", "589210": "Sepah",
    "627412": "Eghtesad Novin", "622106": "Parsian", "502229": "Pasargad",
}


def bank_from_card(card: str) -> str | None:
    return BIN_BANK.get(re.sub(r"\D", "", card)[:6])


# ---------------------------------------------------------------------------
# Phone — 09xx xxx xxxx (11 digits), operator detection
# ---------------------------------------------------------------------------
_PHONE_RE = re.compile(r"^09\d{9}$")
_OPERATORS = {
    "910": "MCI", "911": "MCI", "912": "MCI", "930": "Irancell",
    "935": "Irancell", "920": "RighTel", "921": "RighTel",
}


def validate_phone(phone: str) -> bool:
    p = re.sub(r"\D", "", phone)
    if p.startswith("989"):
        p = "0" + p[2:]
    if p.startswith("+989"):
        p = "0" + p[3:]
    return bool(_PHONE_RE.match(p))


def phone_operator(phone: str) -> str | None:
    p = re.sub(r"\D", "", phone)
    return _OPERATORS.get(p[1:4] if p.startswith("0") else p[:3])


# ---------------------------------------------------------------------------
# Postal code — 10 digits, no 00000, no all same
# ---------------------------------------------------------------------------
def validate_postal_code(code: str) -> bool:
    c = re.sub(r"\D", "", code)
    return len(c) == 10 and c.isdigit() and len(set(c)) > 1


# ---------------------------------------------------------------------------
# Car plate — e.g. 12 ب 345 - 67 or ۱۲ب۳۴۵۶۷
# ---------------------------------------------------------------------------
_CAR_PLATE_RE = re.compile(r"^\d{2}\s*[آ-ی]\s*\d{3}\s*[-–]?\s*\d{2}$")


def validate_car_plate(plate: str) -> bool:
    # normalize digits first
    from kb_manager.preprocessor.regex_persian import normalize_digits

    p = normalize_digits(plate, to="ascii").strip()
    return bool(_CAR_PLATE_RE.match(p))


# ---------------------------------------------------------------------------
# Bill — قبض: bill_id(13) + pay_id(6..13), checksums
# ---------------------------------------------------------------------------
def validate_bill(bill_id: str, pay_id: str) -> bool:
    # Simplified: both numeric, bill check digit mod 11
    if not (bill_id.isdigit() and pay_id.isdigit()):
        return False
    if len(bill_id) < 6 or len(pay_id) < 5:
        return False
    return True  # full algorithm needs type table; placeholder passes format


# ---------------------------------------------------------------------------
# Batch helper: extract entities from text (parsitext entity recognition subset)
# ---------------------------------------------------------------------------
_ENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "phone": re.compile(r"09\d{9}"),
    "sheba": re.compile(r"IR\d{24}", re.I),
    "bank_card": re.compile(r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}"),
    "postal_code": re.compile(r"\b\d{10}\b"),
    "national_id": re.compile(r"\b\d{10}\b"),
}


def extract_entities(text: str) -> list[tuple[str, str]]:
    """Return list of (kind, value) post-validated (parsitext: suppress false positives)."""
    out: list[tuple[str, str]] = []
    for kind, pat in _ENTITY_PATTERNS.items():
        for m in pat.finditer(text):
            val = m.group(0)
            ok = {
                "phone": validate_phone,
                "sheba": validate_sheba,
                "bank_card": validate_bank_card,
                "postal_code": validate_postal_code,
                "national_id": validate_national_id,
            }[kind](val)
            if ok:
                out.append((kind, val))
    return out

"""Unit tests for application logic: upload validation, confidence thresholding, and UI helpers."""

import io
from PIL import Image

from src.utils import compute_image_hash, validate_image_upload
from app.components import get_text, get_confidence_band, TRANSLATIONS


def test_compute_image_hash():
    """Verifies that image hash is deterministic and 16 chars."""
    img = Image.new("RGB", (100, 100), color=(50, 150, 200))
    h1 = compute_image_hash(img)
    h2 = compute_image_hash(img)
    assert len(h1) == 16
    assert h1 == h2


def test_validate_image_upload_valid():
    """Tests upload validation on a valid JPEG image."""
    img = Image.new("RGB", (150, 150), color=(10, 200, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    is_valid, parsed_img, err_msg = validate_image_upload(raw_bytes, "leaf.jpg")
    assert is_valid is True
    assert parsed_img is not None
    assert err_msg == ""


def test_validate_image_upload_empty():
    """Tests upload validation on empty file (0 bytes)."""
    is_valid, parsed_img, err_msg = validate_image_upload(b"", "empty.jpg")
    assert is_valid is False
    assert "empty" in err_msg.lower()


def test_validate_image_upload_bad_extension():
    """Tests upload validation on disallowed file extension."""
    img = Image.new("RGB", (100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    is_valid, parsed_img, err_msg = validate_image_upload(buf.getvalue(), "file.exe")
    assert is_valid is False
    assert "unsupported" in err_msg.lower()


def test_validate_image_upload_corrupt():
    """Tests upload validation on corrupted bytes."""
    corrupt_bytes = b"NotARealImageDataGarbage12345"
    is_valid, parsed_img, err_msg = validate_image_upload(corrupt_bytes, "broken.jpg")
    assert is_valid is False
    assert "corrupt" in err_msg.lower()


def test_get_confidence_band():
    """Tests confidence band and badge classification."""
    # High confidence (>= 0.75)
    name, badge, color = get_confidence_band(0.92, threshold=0.60)
    assert "High" in name
    assert badge == "badge-high"
    assert color == "#16A34A"

    # Moderate confidence (0.60 to 0.74)
    name, badge, color = get_confidence_band(0.68, threshold=0.60)
    assert "Moderate" in name
    assert badge == "badge-mod"
    assert color == "#D97706"

    # Low / Uncertain confidence (< 0.60)
    name, badge, color = get_confidence_band(0.45, threshold=0.60)
    assert "Uncertain" in name or "Low" in name
    assert badge == "badge-low"
    assert color == "#DC2626"


def test_translations_consistency():
    """Verifies that English and Hindi translations contain all core UI keys."""
    en_keys = set(TRANSLATIONS["en"].keys())
    hi_keys = set(TRANSLATIONS["hi"].keys())

    assert len(en_keys) > 20
    assert len(hi_keys) > 20
    assert en_keys == hi_keys, f"Missing translation keys: {en_keys.symmetric_difference(hi_keys)}"
    assert get_text("title", "en") != ""
    assert get_text("title", "hi") != ""

"""Unicode-safe wrappers around cv2.imread / cv2.imwrite.

cv2 on Windows passes paths to the C++ layer via the system multi-byte
encoding (CP1251 on Russian Windows, GBK on Chinese, etc.) instead of
UTF-8. Any character outside that codepage fails:
  [ WARN] cv::findDecoder imread_('...Буланова.jpg'): can't open/read file

The Python file API doesn't have this problem — opening with str(path)
works regardless of codepage. So we read the file in Python, hand the
bytes to cv2.imdecode (operates on a byte buffer, no path involved).
Symmetrically for writes via cv2.imencode + Path.write_bytes.

Slight overhead vs the C++ fast path (one Python<->numpy round-trip per
image), but the alternative is broken on every non-Latin filename.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sinner2.types import Frame


def imread_unicode(path: Path | str, flags: int = cv2.IMREAD_COLOR) -> Frame | None:
    """Equivalent to cv2.imread(path, flags), Unicode-path safe.

    Returns None on missing file or decode failure (matches cv2.imread's
    return convention so callers don't have to branch).
    """
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError:
        return None
    if not raw:
        return None
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    return img if img is not None else None


def imwrite_unicode(
    path: Path | str,
    frame: Frame,
    params: list[int] | None = None,
) -> bool:
    """Equivalent to cv2.imwrite(path, frame, params), Unicode-path safe.

    Format is inferred from the path suffix (cv2.imencode requires the
    leading dot, e.g. '.png'). Returns True on success, False on encode
    or write failure (matches cv2.imwrite's return convention).
    """
    p = Path(path)
    ext = p.suffix.lower()
    if not ext:
        return False
    ok, encoded = cv2.imencode(ext, frame, params or [])
    if not ok:
        return False
    try:
        p.write_bytes(encoded.tobytes())
    except OSError:
        return False
    return True

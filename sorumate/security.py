import time
from collections import defaultdict, deque
from threading import Lock

from . import config
from .errors import ApiError

_lock = Lock()
_hits: dict = defaultdict(deque)


def rate_limit(ip: str) -> None:
    now = time.monotonic()
    with _lock:
        entries = _hits[ip]
        while entries and now - entries[0] > config.RATE_WINDOW_SECONDS:
            entries.popleft()
        if len(entries) >= config.RATE_LIMIT:
            raise ApiError(429, 'ส่งคำขอถี่เกินไป กรุณาลองอีกครั้งภายหลัง')
        entries.append(now)
        if len(_hits) > 2000:
            for key in list(_hits):
                if not _hits[key] or now - _hits[key][-1] > config.RATE_WINDOW_SECONDS:
                    del _hits[key]


def check_access_code(provided) -> None:
    if config.DEMO_ACCESS_CODE and provided != config.DEMO_ACCESS_CODE:
        raise ApiError(403, 'รหัสทดสอบไม่ถูกต้อง')

# A per-employee request quota with the limits library's moving window in memory: one employee
# looping on requests cannot spend the shared model budget. DESKCLERK_REQUESTS_PER_MINUTE sets
# the rate.

from limits import RateLimitItemPerMinute
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter


class Quota:
    def __init__(self, per_minute: int) -> None:
        self._item = RateLimitItemPerMinute(per_minute)
        self._limiter = MovingWindowRateLimiter(MemoryStorage())

    def allows(self, user_id: str) -> bool:
        return self._limiter.hit(self._item, user_id)

class ApiError(Exception):
    """Framework-free error. main.py turns it into the same JSON shape as
    FastAPI's HTTPException: {"detail": "..."}."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail

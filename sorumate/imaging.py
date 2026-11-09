import io

from PIL import Image, ImageOps, UnidentifiedImageError

from . import config
from .errors import ApiError


def load_image(data: bytes) -> Image.Image:
    """Validate uploaded bytes and return an upright RGB image (in memory only)."""
    if not data:
        raise ApiError(400, 'ไม่มีข้อมูลภาพ')
    if len(data) > config.MAX_BYTES:
        raise ApiError(413, 'ภาพต้องมีขนาดไม่เกิน 10 MB')
    try:
        with Image.open(io.BytesIO(data)) as original:
            if original.width * original.height > config.MAX_PIXELS:
                raise ApiError(413, 'ภาพต้องมีขนาดไม่เกิน 20 ล้านพิกเซล')
            original.load()
            return ImageOps.exif_transpose(original).convert('RGB')
    except ApiError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise ApiError(400, 'กรุณาใช้รูป JPG, PNG หรือ WebP ที่เปิดได้')

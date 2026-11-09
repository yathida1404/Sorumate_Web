"""Run the ONNX model on one image and print what the backend would return.

    python scripts/try_onnx.py photo.jpg [models/best.onnx]

Use this once with a real photo to confirm the output format and class names.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image, ImageOps  # noqa: E402
from sorumate import config  # noqa: E402
from sorumate.inference.common import build_response  # noqa: E402
from sorumate.inference.onnx_engine import OnnxEngine  # noqa: E402

if len(sys.argv) < 2:
    sys.exit(__doc__)
model = Path(sys.argv[2]) if len(sys.argv) > 2 else config.ONNX_MODEL_PATH
eng = OnnxEngine(model)
print('input size (h, w):', eng.size, '| classes:', len(eng.names))
img = ImageOps.exif_transpose(Image.open(sys.argv[1])).convert('RGB')
print(json.dumps(build_response(eng.detect(img), 'onnx'), ensure_ascii=False, indent=2))

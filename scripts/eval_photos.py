"""Measure detection on YOUR OWN photos, per class.

Put real photos (phone, dorm lighting, real backgrounds) in folders named by model class:
    photos/PET Bottle/001.jpg
    photos/Paper Cup/xyz.jpg
    photos/_background/empty_table.jpg     (no waste: should detect nothing)

Run:
    python scripts/eval_photos.py photos            # uses models/best.onnx
    python scripts/eval_photos.py photos --thr 0.2  # try another threshold

Prints, per class: hit rate (right class found), miss (nothing found), and what it was confused with.
Use the worst classes to decide which photos to add to the training set next.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sorumate import config  # noqa: E402
from sorumate.catalog import normal  # noqa: E402
from sorumate.imaging import load_image  # noqa: E402
from sorumate.inference.onnx_engine import OnnxEngine  # noqa: E402

EXT = {'.jpg', '.jpeg', '.png', '.webp'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--model', default=str(config.ONNX_MODEL_PATH))
    ap.add_argument('--thr', type=float, default=config.THRESHOLD)
    ap.add_argument('--resize', choices=['letterbox', 'stretch'], default=config.RESIZE_MODE,
                    help='try both: the one with the higher hit rate matches your training data')
    args = ap.parse_args()
    config.THRESHOLD = config.CANDIDATE_THRESHOLD = min(args.thr, config.CANDIDATE_THRESHOLD)
    config.RESIZE_MODE = args.resize
    engine = OnnxEngine(Path(args.model))
    stats = defaultdict(Counter)
    for cls_dir in sorted(p for p in Path(args.folder).iterdir() if p.is_dir()):
        truth = cls_dir.name
        for img in sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in EXT):
            dets = [d for d in engine.detect(load_image(img.read_bytes())) if d.confidence >= args.thr]
            names = [d.name for d in sorted(dets, key=lambda d: -d.confidence)]
            s = stats[truth]
            s['n'] += 1
            if truth == '_background':
                s['hit' if not names else 'false_alarm'] += 1
                for n in names:
                    s['as:' + n] += 1
            elif any(normal(n) == normal(truth) for n in names):
                s['hit'] += 1
            elif not names:
                s['miss'] += 1
            else:
                s['wrong'] += 1
                s['as:' + names[0]] += 1
    print(f'threshold={args.thr}  resize={args.resize}')
    print(f'{"class":28} {"n":>4} {"hit%":>6} {"miss":>5} {"wrong":>6}  confused with')
    for truth, s in sorted(stats.items(), key=lambda kv: kv[1]['hit'] / max(1, kv[1]['n'])):
        conf = ', '.join(f'{k[3:]}×{v}' for k, v in s.most_common() if k.startswith('as:'))[:60]
        print(f'{truth:28} {s["n"]:>4} {100 * s["hit"] / max(1, s["n"]):>5.0f}% {s["miss"]:>5} {s["wrong"] + s["false_alarm"]:>6}  {conf}')


if __name__ == '__main__':
    main()

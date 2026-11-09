"""Framework-free checks. Run:  python tests/test_core.py"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sorumate import catalog as cat  # noqa: E402
from sorumate.errors import ApiError  # noqa: E402
from sorumate.imaging import load_image  # noqa: E402
from sorumate.inference import common, onnx_engine as oe  # noqa: E402


def test_catalog_and_missing_extras():
    items = cat.catalog()['items']
    assert len(items) == 25
    assert cat.extra_items() == [] or isinstance(cat.extra_items(), list)
    assert 'outside_items' in cat.full_catalog()


def test_all_25_classes_resolve():
    for c in cat.catalog()['model_classes']:
        label = common.resolve_label(c)[0]
        assert label == cat.resolve_model_class(c), c
        assert cat.get_item(label), f'{c} -> {label} not in catalog'


def test_ask():
    r = cat.ask({'item_id': 'battery', 'answers': {'battery_state': 'intact'}})
    assert r['found'] and r['category_id'] == 'hazardous' and not r['out_of_25']
    r = cat.ask({'query': 'definitely not a real item'})
    assert r['found'] is False and r['out_of_25'] is True
    r = cat.ask({'item_id': 'glass_bottle', 'answers': {'glass_state': 'broken'}})
    assert r['category_id'] == 'nonburnable'
    r = cat.ask({'item_id': 'light_bulb'})
    assert r['category_id'] == 'review' and r['requires_confirmation']
    ready_pet = cat.ask({'item_id': 'pet_plastic_bottle', 'answers': {'pet_kind':'eligible', 'pet_cap':'removed', 'pet_label':'removed', 'pet_clean':'clean'}})
    assert ready_pet['category_id'] == 'pet' and ready_pet['ready'] and len(ready_pet['steps']) >= 3
    capped_pet = cat.ask({'item_id': 'pet_plastic_bottle', 'answers': {'pet_kind':'eligible', 'pet_cap':'attached', 'pet_label':'removed', 'pet_clean':'clean'}})
    assert capped_pet['category_id'] == 'pet' and not capped_pet['ready']
    assert 'ฝา' in capped_pet['message_th']
    assert all('PET' not in x.get('en','') for x in cat.ask({'item_id':'glass_bottle','answers':{'glass_state':'broken'}})['steps'])


def test_letterbox_roundtrip():
    img = Image.new('RGB', (400, 200))
    t, r, px, py = oe.letterbox(img, (640, 640))
    assert t.shape == (1, 3, 640, 640) and abs(r - 1.6) < 1e-6 and px == 0 and py == 160


def test_decode_end2end():
    out = np.array([[[10, 20, 110, 220, 0.9, 3], [0, 0, 5, 5, 0.1, 1]]], dtype=np.float32)
    b, s, c = oe.decode(out, 0.3, 0.7)
    assert len(b) == 1 and c[0] == 3 and abs(s[0] - 0.9) < 1e-6


def test_decode_classic_with_nms():
    nc, n = 3, 50   # real models have thousands of anchors; N >> 4+nc
    out = np.zeros((1, 4 + nc, n), np.float32)
    # two overlapping boxes of class 1 (keep the stronger), one far box class 2
    out[0, :, 0] = [100, 100, 80, 80, 0.0, 0.9, 0.0]
    out[0, :, 1] = [102, 101, 80, 80, 0.0, 0.7, 0.0]
    out[0, :, 2] = [400, 400, 50, 50, 0.0, 0.0, 0.8]
    out[0, :, 3] = [300, 300, 50, 50, 0.1, 0.1, 0.1]
    b, s, c = oe.decode(out, 0.3, 0.7)
    assert sorted(c.tolist()) == [1, 2] and len(b) == 2


def test_agnostic_nms_removes_duplicate_across_classes():
    boxes = np.array([[10, 10, 200, 200], [12, 11, 198, 205], [400, 400, 450, 450]], np.float32)
    scores = np.array([0.55, 0.40, 0.8], np.float32)
    cls = np.array([20, 3, 5])
    b, s, c = oe.agnostic_nms(boxes, scores, cls, 0.6)
    assert sorted(c.tolist()) == [5, 20] and len(b) == 2          # the 0.40 duplicate (class 3) is gone
    b, s, c = oe.agnostic_nms(boxes, scores, cls, 1.0)            # 1.0 = disabled
    assert len(b) == 3


class _FakeSess:
    def __init__(self, out): self._out = out
    def run(self, _n, feed): assert feed['images'].shape == (1, 3, 640, 640); return [self._out]


def test_engine_maps_back_to_original_image():
    """640x640 letterboxed box on a 400x200 image must come back normalized to the ORIGINAL."""
    eng = object.__new__(oe.OnnxEngine)
    eng.input_name, eng.size = 'images', (640, 640)
    eng.names = ['Battery', 'PET Bottle']
    # in the 640 canvas the image occupies y 160..480; box covers left half of the image
    eng.sess = _FakeSess(np.array([[[0, 160, 320, 480, 0.88, 1]]], np.float32))
    d = eng.detect(Image.new('RGB', (400, 200)))[0]
    assert d.name == 'PET Bottle'
    assert (round(d.x1, 3), round(d.y1, 3), round(d.x2, 3), round(d.y2, 3)) == (0.0, 0.0, 0.5, 1.0)
    resp = common.build_response([d], 'onnx')
    assert resp['detections'][0]['label'] == 'pet_plastic_bottle'
    assert resp['detections'][0]['bbox'] == [0.0, 0.0, 0.5, 1.0]


def test_build_response_threshold_and_sort():
    raw = [common.RawDet('Battery', 0.2, 0, 0, .1, .1), common.RawDet('Lighter', 0.5, 0, 0, .2, .2),
           common.RawDet('Battery', 0.9, .1, .1, .3, .3)]
    r = common.build_response(raw, 'x', 0.3)
    assert [d['confidence'] for d in r['detections']] == [0.9, 0.5]


def test_load_image_errors():
    for bad, code in [(b'', 400), (b'not an image', 400)]:
        try:
            load_image(bad); assert False
        except ApiError as e:
            assert e.status == code


if __name__ == '__main__':
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); n += 1; print('ok', name)
    print(f'{n} tests passed')

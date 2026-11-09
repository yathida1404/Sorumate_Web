"""Waste-rule data and lookup. Pure Python (no web framework) so it can be tested alone.

Data files:
  data/waste_catalog.json      the 25 model classes + verified rules
  data/other_waste_items.json  optional extra searchable items (outside the 25)
"""
import json
import logging
import re
import unicodedata
from functools import lru_cache

from . import config

log = logging.getLogger('sorumate')


@lru_cache(maxsize=1)
def catalog() -> dict:
    with open(config.DATA_DIR / 'waste_catalog.json', encoding='utf-8') as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def extra_items() -> list:
    """Items outside the 25 classes. A missing file is treated as an empty list
    (the original code crashed with HTTP 500 here)."""
    path = config.DATA_DIR / 'other_waste_items.json'
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh).get('items', [])
    except FileNotFoundError:
        log.warning('%s not found; no extra items loaded', path)
        return []


def normal(text) -> str:
    return unicodedata.normalize('NFKC', str(text or '')).casefold().strip().replace('_', ' ').replace('-', ' ')


@lru_cache(maxsize=1)
def by_name() -> dict:
    entries = {}
    for item in catalog()['items']:
        for x in [item['id'], item['en'], item['th'], item['ja'], item.get('model_class', ''), *item.get('aliases', [])]:
            if x:
                entries.setdefault(normal(x), item)
    # Reserve exact model-class matches so broad aliases cannot overwrite them.
    for item in catalog()['items']:
        entries[normal(item['model_class'])] = item
    return entries


def resolve_model_class(name) -> str:
    item = by_name().get(normal(name))
    return item['id'] if item else (re.sub(r'[^a-z0-9]+', '_', normal(name)).strip('_') or 'unknown')


def get_item(item_id: str):
    return next((x for x in catalog()['items'] if x['id'] == item_id), None)


def model_classes_info() -> dict:
    d = catalog()
    return {'count': len(d.get('model_classes', [])), 'classes': d.get('model_classes', []),
            'class_to_item_id': d.get('model_class_to_item_id', {}),
            'note_th': d.get('model_mapping_notice_th', '')}


def full_catalog() -> dict:
    # Keep the trained 25-class list separate from extra searchable guide items.
    return {**catalog(), 'outside_items': extra_items()}


def choose_category(item: dict, answers: dict) -> str:
    a = answers or {}
    kind = item['category']
    if item['id'] in ('cup', 'paper_cup', 'plastic_cup'):
        kind = {'paper': 'burnable', 'plastic_soft': 'burnable', 'plastic_hard': 'nonburnable'}.get(a.get('material'), 'review')
    elif item['id'] == 'light_bulb':
        kind = {'led': 'nonburnable', 'fluorescent': 'hazardous'}.get(a.get('bulb_type'), 'review')
    elif item['id'] == 'glass_bottle':
        kind = {'intact': 'recyclables', 'broken': 'nonburnable'}.get(a.get('state'), 'review')
    return kind


def ask(payload: dict) -> dict:
    """Exact catalog/guide retrieval only. Unrecognized text is never classified by an LLM."""
    query = str(payload.get('query') or '').strip()[:160]
    explicit_id = str(payload.get('item_id') or '').strip()
    answers = payload.get('answers') if isinstance(payload.get('answers'), dict) else {}
    items = catalog()['items']
    extras = extra_items()
    match = next((x for x in [*items, *extras] if explicit_id and x['id'] == explicit_id), None)
    if match is None and query:
        # Do not silently choose between ambiguous matches (e.g., cup vs glass bottle).
        options = [i for i in [*items, *extras]
                   if any(normal(query) == normal(x) for x in [i['en'], i['th'], i['ja'], *i.get('aliases', [])])]
        if len(options) == 1:
            match = options[0]
    if match is None:
        return {
            'found': False,
            'reason': 'unverified_item',
            'query': query,
            'message_th': 'ยังไม่มีข้อมูลที่ยืนยันวิธีทิ้งสำหรับสิ่งของนี้ โปรดตรวจคู่มือแยกขยะเมืองเอเบ็ตสึ หรือสอบถาม 011-384-5600',
            'message_ja': 'この品目の分別方法は確認できません。江別市の公式手引きを確認するか、011-384-5600へお問い合わせください。',
            'message_en': 'No verified disposal rule is available for this item. Check the official Ebetsu guide or call 011-384-5600.',
            'source': catalog()['sources'][0],
            'requires_confirmation': True,
            'out_of_25': True,
            'suggestions': [],
        }
    in_25 = any(i['id'] == match['id'] for i in items)
    kind = choose_category(match, answers) if in_25 else match['category']
    return {
        'found': True,
        'item': match,
        'category_id': kind,
        'category': catalog()['category_names'][kind],
        'requires_confirmation': kind == 'review' or match.get('status') in ('conditional', 'provisional'),
        'message_th': match['details_th'],
        'message_ja': match['details_ja'],
        'message_en': match['details_en'],
        'out_of_25': not in_25,
        'source': match['source'],
    }

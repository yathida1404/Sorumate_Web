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


def _matches(answers: dict, conditions: dict) -> bool:
    """All condition fields must match; a list means one of the stated values."""
    return all(
        answers.get(key) in value if isinstance(value, list) else answers.get(key) == value
        for key, value in conditions.items()
    )


def active_questions(item: dict, answers: dict) -> list:
    return [q for q in item.get('questions', []) if _matches(answers, q.get('when', {}))]


def choose_guidance(item: dict, answers: dict) -> dict:
    """Resolve a pre-authored city rule. Missing answers never imply readiness.

    Both scan and typed-search clients use this same decision via /api/ask.
    """
    rules = item.get('disposal')
    if not rules:
        # Backwards compatibility with older additional-items JSON records.
        category = item.get('category', 'review')
        return {'category': category, 'ready': category != 'review' and item.get('status') == 'validated',
                'steps': [{'icon': '📋', 'th': item.get('details_th', ''),
                           'ja': item.get('details_ja', ''), 'en': item.get('details_en', '')}]}
    if any(q.get('id') not in answers or answers[q['id']] not in {opt['id'] for opt in q['options']}
           for q in active_questions(item, answers)):
        return {'category': 'review', 'ready': False, 'steps': [
            {'icon': '🔎', 'th': 'กรุณาตอบคำถามเรื่องวัสดุและสภาพให้ครบก่อนดูวิธีทิ้ง',
             'en': 'Answer all material and condition questions before disposal.',
             'ja': '分別の前に材質・状態についてすべて回答してください。'}]}
    for case in rules.get('cases', []):
        if _matches(answers, case.get('when', {})):
            return case
    return rules['default']


def choose_category(item: dict, answers: dict) -> str:
    return choose_guidance(item, answers or {})['category']


def _s(icon, th, ja, en):
    return {'icon': icon, 'th': th, 'ja': ja, 'en': en}


_EST_NOTE = _s('🔎', 'คำตอบนี้ประเมินจากวัสดุ/สภาพที่คุณเลือกตามหลักของเมือง ไม่ใช่รายการที่ยืนยันชื่อแล้ว',
               '選択した材質・状態から市の基準で推定した結果です（品目名での確認ではありません）。',
               'Estimated from the material/condition you chose using the city\'s general rules, not a confirmed item entry.')
_BAG_B = _s('🛍️', 'ใส่ถุงขยะเผาได้ที่เมืองกำหนด นำออกวันเก็บขยะเผาได้', '市指定の燃やせるごみ袋で、収集日に出す。',
            'Use the city burnable-waste bag on burnable collection day.')
_BAG_N = _s('🛍️', 'ใส่ถุงขยะเผาไม่ได้ที่เมืองกำหนด นำออกวันเก็บขยะเผาไม่ได้', '市指定の燃やせないごみ袋で、収集日に出す。',
            'Use the city non-burnable-waste bag on non-burnable collection day.')


def estimate_from_material(query: str, material: str, condition: str):
    """Fallback for items not in either JSON: apply Ebetsu's general category rules
    (72940.pdf / 71884.pdf) to the material the user chose. Never used without a material."""
    m, c = material.strip(), condition.strip()
    if m == 'paper':
        kind, steps = 'burnable', [_s('📄', 'กระดาษทุกชนิด รวมกระดาษที่เปื้อนอาหาร เป็นขยะเผาได้ (กระดาษสะอาดส่งเก็บกระดาษของชุมชนได้)',
                                       '紙類は汚れていても燃やせるごみ（きれいな紙は集団資源回収へ）。',
                                       'All paper, including food-stained paper, is burnable (clean paper can go to community paper collection).'),
                                   _s('🥛', 'ยกเว้นกล่องนมกระดาษด้านในสีขาว = ขยะรีไซเคิล (紙パック)', '内側が白い紙パックは資源物。',
                                      'Exception: paper drink cartons with a white inside are recyclables (紙パック).'), _BAG_B]
    elif m == 'soft_plastic':
        kind, steps = 'burnable', [_s('🔥', 'พลาสติกบาง/นิ่มเป็นขยะเผาได้ เปื้อนก็ทิ้งได้', '薄い・軟らかいプラスチックは燃やせるごみ。',
                                       'Thin/soft plastic is burnable, even if dirty.'), _BAG_B]
    elif m == 'hard_plastic':
        kind, steps = 'burnable', [_s('🔥', 'พลาสติกแข็งทั่วไปเป็นขยะเผาได้ ยกเว้นก้อนหนาเกิน 2 ซม. = ขยะเผาไม่ได้',
                                       '硬いプラスチックも燃やせるごみ。厚さ2cm超のかたまりは燃やせないごみ。',
                                       'Rigid plastic is burnable too; only solid lumps over 2 cm thick are non-burnable.'), _BAG_B]
    elif m == 'glass':
        kind, steps = 'nonburnable', [_s('🧱', 'แก้ว กระจก เซรามิก เป็นขยะเผาไม่ได้' + (' ห่อของแตกด้วยกระดาษ' if c == 'broken' else ''),
                                          'ガラス・陶磁器は燃やせないごみ。' + ('割れたものは紙で包む。' if c == 'broken' else ''),
                                          'Glass and ceramics are non-burnable.' + (' Wrap broken pieces in paper.' if c == 'broken' else ''))]
        if c != 'broken':
            steps.append(_s('🍾', 'ยกเว้นขวดแก้วเครื่องดื่ม/อาหาร/เครื่องสำอางที่ไม่แตก = ขยะรีไซเคิล (ค้นหา "ขวดแก้ว")',
                            '割れていない飲食用・化粧品のびんは資源物。', 'Exception: unbroken food/drink/cosmetic bottles are recyclables (search "glass bottle").'))
        steps.append(_BAG_N)
    elif m == 'metal':
        kind, steps = 'nonburnable', [_s('🔩', 'ของโลหะเป็นขยะเผาไม่ได้', '金属類は燃やせないごみ。', 'Metal items are non-burnable.'),
                                      _s('🥫', 'ยกเว้นกระป๋องอาหาร/เครื่องดื่มที่ล้างแล้ว = รีไซเคิล; สเปรย์/แก๊สกระป๋อง = ขยะอันตราย',
                                         '飲食用の缶は資源物、スプレー缶・ガス缶は危険ごみ。',
                                         'Exceptions: rinsed food/drink cans are recyclables; spray cans and gas cartridges are hazardous.'), _BAG_N]
    elif m == 'electrical':
        kind, steps = 'review', [_s('🔋', 'ถอดถ่าน/แบตเตอรี่ออกก่อน (แบตเป็นขยะอันตราย)', '電池は外して危険ごみへ。',
                                     'Remove batteries first (batteries are hazardous waste).'),
                                  _s('🔌', 'เครื่องใช้ไฟฟ้าขนาดเล็กมีการเก็บแยก (小型家電) ตรวจคู่มือหรือโทร 011-384-5600',
                                     '小型家電は別回収。手引きで確認。', 'Small appliances have a separate collection (小型家電). Check the guide or call 011-384-5600.')]
    else:
        return None
    name = query or {'paper': 'กระดาษ', 'soft_plastic': 'พลาสติกนิ่ม', 'hard_plastic': 'พลาสติกแข็ง',
                     'glass': 'แก้ว', 'metal': 'โลหะ', 'electrical': 'เครื่องใช้ไฟฟ้า'}[m]
    steps = [_EST_NOTE, *steps]
    return {
        'found': True, 'estimated': True,
        'item': {'id': f'estimate_{m}', 'th': name, 'ja': name, 'en': name, 'questions': []},
        'category_id': kind, 'category': catalog()['category_names'][kind],
        'ready': False, 'requires_confirmation': True, 'steps': steps,
        'message_th': ' '.join(s['th'] for s in steps), 'message_ja': ' '.join(s['ja'] for s in steps),
        'message_en': ' '.join(s['en'] for s in steps),
        'out_of_25': True, 'source': 'https://www.city.ebetsu.hokkaido.jp/uploaded/attachment/72940.pdf',
    }


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
        est = estimate_from_material(query, str(payload.get('material') or ''), str(payload.get('condition') or ''))
        if est:
            return est
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
    guidance = choose_guidance(match, answers)
    kind = guidance['category']
    if kind not in catalog()['category_names']:
        kind = 'review'
        guidance = {'category': kind, 'ready': False, 'steps': []}
    ready = bool(guidance.get('ready', False)) and kind != 'review'
    steps = guidance.get('steps', [])
    return {
        'found': True,
        'item': match,
        'category_id': kind,
        'category': catalog()['category_names'][kind],
        'ready': ready,
        'requires_confirmation': not ready,
        'steps': steps,
        'message_th': ' '.join(s.get('th', '') for s in steps) or match.get('details_th', ''),
        'message_ja': ' '.join(s.get('ja', '') for s in steps) or match.get('details_ja', ''),
        'message_en': ' '.join(s.get('en', '') for s in steps) or match.get('details_en', ''),
        'out_of_25': not in_25,
        'source': match.get('source', catalog()['sources'][0]),
    }

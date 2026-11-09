"""Compare the trained model's class NAMES with data/waste_catalog.json.

    python scripts/check_classes.py data.yaml
    python scripts/check_classes.py models/best.onnx

The backend matches classes by NAME (case/underscore-insensitive), not by ID,
so what matters is that every model class name has a catalog entry.
"""
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sorumate.catalog import by_name, catalog, normal  # noqa: E402


def names_from_yaml(path):
    text = Path(path).read_text(encoding='utf-8')
    try:
        import yaml
        n = yaml.safe_load(text)['names']
        return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
    except ImportError:
        m = re.search(r'names\s*:\s*(\[.*?\])', text, re.S)
        if not m:
            sys.exit('Install pyyaml (pip install pyyaml) or use a names: [...] list')
        return ast.literal_eval(m.group(1))


def names_from_onnx(path):
    import onnxruntime as ort
    from sorumate.inference.onnx_engine import parse_names
    meta = ort.InferenceSession(path, providers=['CPUExecutionProvider']).get_modelmeta().custom_metadata_map
    return parse_names(meta.get('names'))


if len(sys.argv) != 2:
    sys.exit(__doc__)
src = sys.argv[1]
model = names_from_onnx(src) if src.endswith('.onnx') else names_from_yaml(src)
cat_classes = catalog().get('model_classes', [])

print(f'Model classes: {len(model)} | Catalog model_classes: {len(cat_classes)}')
missing = [n for n in model if normal(n) not in by_name()]
unused = [c for c in cat_classes if normal(c) not in {normal(n) for n in model}]
print('\nModel classes with NO catalog entry (will show as plain name, no rule):')
print('  ' + (', '.join(missing) if missing else 'none'))
print('\nCatalog classes the model does NOT have (rules that can never be triggered by AI):')
print('  ' + (', '.join(unused) if unused else 'none'))
print('\nOK - all matched' if not missing and not unused else '\nFix names in data/waste_catalog.json (model_class / aliases) or retrain.')

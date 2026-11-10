"""Compare original/inplace/lookup FP32 preprocessing bytes on an explicit split.

Requires numpy and Pillow plus sibling preprocess_candidates.py and common.py.
Split rows use the existing first tab-separated column as the image path.
"""
from pathlib import Path
import argparse
import hashlib
import json

def check(split, data_root):
    from preprocess_candidates import METHODS
    rows = [line.split('\t')[0] for line in split.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('The split is empty.')
    for i, name in enumerate(rows):
        path = (data_root / name).resolve()
        if not path.is_relative_to(data_root) or not path.is_file():
            raise ValueError('Missing image or path outside --data-root: ' + name)
        reference = METHODS['original'](path)
        for mode in ('inplace', 'lookup'):
            output = METHODS[mode](path)
            if output.shape != reference.shape or output.dtype != reference.dtype or output.tobytes() != reference.tobytes():
                raise AssertionError('Preprocessing bytes differ: ' + mode + ' ' + name)
        if (i + 1) % 300 == 0:
            print('pixel checks', i + 1, flush=True)
    return dict(n=len(rows), methods=['inplace', 'lookup'], all_fp32_input_bytes_identical=True, split=str(split), split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(), data_root=str(data_root))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', required=True, type=Path, help='UTF-8 split file; first tab-separated column contains image paths.')
    parser.add_argument('--data-root', required=True, type=Path, help='Directory against which split image paths are resolved.')
    parser.add_argument('--output', required=True, type=Path, help='New JSON result file.')
    args = parser.parse_args()
    split, data_root, output = (args.split.resolve(), args.data_root.resolve(), args.output.resolve())
    if not split.is_file() or not data_root.is_dir():
        parser.error('--split must be a file and --data-root must be a directory.')
    if output.exists():
        parser.error('Use a new --output path; refusing to overwrite: ' + str(output))
    report = check(split, data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('ALL_VALIDATION_INPUTS_BYTE_IDENTICAL', report['n'], flush=True)
if __name__ == '__main__':
    main()

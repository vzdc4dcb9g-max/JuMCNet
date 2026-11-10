"""Fuse seven JuMCNet depthwise Conv + SiLU blocks using jumc.cpu.

Requires numpy and onnx. Inference requires the matching ARM dw_silu library;
ONNX checker validation alone is not a numerical equivalence test.
Optional fixtures are generated only in an explicitly requested new directory.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json

def fuse(source, output):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    source, output = (Path(source), Path(output))
    m = onnx.load(str(source))
    orig = copy.deepcopy(m)
    if not len({n.name for n in m.graph.node}) == len(m.graph.node):
        raise ValueError('Node names must be unique')
    graph_outputs = {v.name for v in m.graph.output}
    cons = {}
    for n in m.graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    weights = {w.name: numpy_helper.to_array(w) for w in m.graph.initializer}
    remove = set()
    replace = {}
    records = []
    for n in m.graph.node:
        if n.op_type != 'Conv' or len(n.input) != 3:
            continue
        a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
        w = weights[n.input[1]]
        if w.ndim != 4 or w.shape[1:] != (1, 3, 3) or a.get('group') != w.shape[0] or (a.get('pads') != [1] * 4) or (a.get('dilations') != [1, 1]) or (a.get('strides') not in [[1, 1], [2, 2]]):
            continue
        uses = cons.get(n.output[0], [])
        if len(uses) != 2:
            continue
        sig = next((x for x in uses if x.op_type == 'Sigmoid'), None)
        mul = next((x for x in uses if x.op_type == 'Mul'), None)
        if sig is None or mul is None or set(mul.input) != {n.output[0], sig.output[0]} or (len(cons.get(sig.output[0], [])) != 1):
            continue
        if not (w.dtype == np.float32 and weights[n.input[2]].dtype == np.float32):
            raise ValueError('Depthwise weights and bias must be FP32')
        if not weights[n.input[2]].shape == (w.shape[0],):
            raise ValueError('Depthwise bias shape mismatch')
        if not (list(sig.input) == list(n.output) and len(n.output) == 1 and (len(sig.output) == 1) and (len(mul.output) == 1)):
            raise ValueError('Graph contract check failed: list(sig.input) == list(n.output) and len(n.output) == 1 and (len(sig.output) == 1) and (len(mul.output) == 1)')
        if not not (set(n.output) | set(sig.output)) & graph_outputs:
            raise ValueError('Internal depthwise value is a graph output')
        if not (a.get('auto_pad', b'NOTSET') == b'NOTSET' and a.get('kernel_shape', [3, 3]) == [3, 3]):
            raise ValueError("Graph contract check failed: a.get('auto_pad', b'NOTSET') == b'NOTSET' and a.get('kernel_shape', [3, 3]) == [3, 3]")
        r = helper.make_node('DepthwiseConvSilu', list(n.input), list(mul.output), name=n.name + '/FusedSilu', domain='jumc.cpu', stride=a['strides'][0])
        replace[n.name] = r
        remove.update([sig.name, mul.name])
        records.append(dict(name=n.name, stride=a['strides'][0], channels=w.shape[0], weight_sha256=hashlib.sha256(w.tobytes()).hexdigest()))
    nodes = [replace.get(n.name, n) for n in m.graph.node if n.name not in remove]
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    m.opset_import.append(helper.make_opsetid('jumc.cpu', 1))
    if not len(records) == 7:
        raise ValueError('Graph contract check failed: len(records) == 7')
    if not [x.SerializeToString() for x in orig.graph.initializer] == [x.SerializeToString() for x in m.graph.initializer]:
        raise ValueError('Graph contract check failed: [x.SerializeToString() for x in orig.graph.initializer] == [x.SerializeToString() for x in m.graph.initializer]')
    if not [x.SerializeToString() for x in orig.graph.node if x.op_type == 'Attention'] == [x.SerializeToString() for x in m.graph.node if x.op_type == 'Attention']:
        raise ValueError("Graph contract check failed: [x.SerializeToString() for x in orig.graph.node if x.op_type == 'Attention'] == [x.SerializeToString() for x in m.graph.node if x.op_type == 'Attention']")
    onnx.checker.check_model(m)
    onnx.save(m, str(output))
    return dict(fused_nodes=records, count=len(records), source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), output=str(output), output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), initializers_byte_identical=True, attention_nodes_unchanged=True, contract='FP32 NEON 3x3 depthwise Conv plus SiLU, one output pass, ORT thread pool', numerical_validation='not_run; requires ARM custom operator library')

def make_fixtures(directory):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(1717)
    for s, h in [(1, 56), (2, 112), (1, 7), (2, 7)]:
        c = 32
        x = rng.normal(0, 2, (1, c, h, h)).astype('float32')
        w = rng.normal(0, 0.4, (c, 1, 3, 3)).astype('float32')
        b = rng.normal(0, 0.4, (c,)).astype('float32')
        init = [numpy_helper.from_array(w, 'w'), numpy_helper.from_array(b, 'b')]
        shape = [1, c, (h - 1) // s + 1, (h - 1) // s + 1]
        for mode in ['original', 'fused']:
            ns = [helper.make_node('Conv', ['x', 'w', 'b'], ['z'], group=c, kernel_shape=[3, 3], pads=[1] * 4, strides=[s, s]), helper.make_node('Sigmoid', ['z'], ['sig']), helper.make_node('Mul', ['z', 'sig'], ['y'])] if mode == 'original' else [helper.make_node('DepthwiseConvSilu', ['x', 'w', 'b'], ['y'], domain='jumc.cpu', stride=s)]
            mm = helper.make_model(helper.make_graph(ns, 'test', [helper.make_tensor_value_info('x', 1, list(x.shape))], [helper.make_tensor_value_info('y', 1, shape)], init), opset_imports=[helper.make_opsetid('', 17), helper.make_opsetid('jumc.cpu', 1)], ir_version=9)
            onnx.save(mm, directory / f'test_s{s}_h{h}_{mode}.onnx')
        np.save(directory / f'test_s{s}_h{h}_input.npy', x)

def output_paths(parser, source, output, report):
    source, output = (source.resolve(), output.resolve())
    report = (report or output.with_suffix('.json')).resolve()
    if not source.is_file():
        parser.error('Source model is missing: ' + str(source))
    if len({source, output, report}) != 3:
        parser.error('Source model, output model, and report must have distinct paths.')
    for path in (output, report):
        if path.exists():
            parser.error('Use a new output path; refusing to overwrite: ' + str(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    return (source, output, report)

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Attention-fused JuMCNet FP32 ONNX graph.')
    parser.add_argument('--output', required=True, type=Path, help='New depthwise-fused ONNX path.')
    parser.add_argument('--report', type=Path, help='New JSON path; defaults to OUTPUT with .json suffix.')
    parser.add_argument('--fixtures-dir', type=Path, help='Generate original/fused branch fixture models and random inputs here; does not run them.')
    parser.add_argument('--no-tests', action='store_true', help='Compatibility flag: skip optional fixture generation (the default).')
    args = parser.parse_args()
    if args.no_tests and args.fixtures_dir:
        parser.error('--no-tests and --fixtures-dir cannot be combined.')
    if args.fixtures_dir:
        args.fixtures_dir = args.fixtures_dir.resolve()
        if args.fixtures_dir.exists():
            parser.error('Use a new --fixtures-dir; refusing to overwrite old fixtures.')
        for path in (args.output.resolve(), (args.report or args.output.with_suffix('.json')).resolve()):
            if path == args.fixtures_dir or path.is_relative_to(args.fixtures_dir):
                parser.error('--fixtures-dir must be separate from the output model/report paths.')
    source, output, report_path = output_paths(parser, args.source, args.output, args.report)
    report = fuse(source, output)
    if args.fixtures_dir:
        make_fixtures(args.fixtures_dir)
        report['fixtures'] = dict(directory=str(args.fixtures_dir), generated=True, executed=False)
    write_json(report_path, report)
    print('FUSED', report['count'], output, flush=True)
if __name__ == '__main__':
    main()

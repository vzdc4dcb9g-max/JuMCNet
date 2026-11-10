"""Fuse the two fixed JuMCNet 16-to-32 pointwise Conv + SiLU blocks.

Requires onnx and numpy. Input must already contain nine Attention and seven
DepthwiseConvSilu nodes. Inference also needs the matching ARM custom libraries.
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
    targets = ['/stages.0/stages.0.0/conv1_1x1/conv/Conv', '/stages.1/stages.1.0/conv1_1x1/conv/Conv']
    m = onnx.load(str(source))
    old = copy.deepcopy(m)
    if not len({n.name for n in m.graph.node}) == len(m.graph.node):
        raise ValueError('Node names must be unique')
    graph_outputs = {v.name for v in m.graph.output}
    weights = {w.name: numpy_helper.to_array(w) for w in m.graph.initializer}
    cons = {}
    for n in m.graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    replacements, remove, records = ({}, set(), [])
    for n in m.graph.node:
        if n.name not in targets:
            continue
        attrs = {x.name: helper.get_attribute_value(x) for x in n.attribute}
        if not (n.op_type == 'Conv' and len(n.input) == 3 and (weights[n.input[1]].shape == (32, 16, 1, 1))):
            raise ValueError("Graph contract check failed: n.op_type == 'Conv' and len(n.input) == 3 and (weights[n.input[1]].shape == (32, 16, 1, 1))")
        if not (weights[n.input[1]].dtype == np.float32 and weights[n.input[2]].dtype == np.float32):
            raise ValueError('Pointwise weights and bias must be FP32')
        if not weights[n.input[2]].shape == (32,):
            raise ValueError('Pointwise bias shape mismatch')
        if not (attrs.get('auto_pad', b'NOTSET') == b'NOTSET' and attrs.get('kernel_shape', [1, 1]) == [1, 1]):
            raise ValueError("Graph contract check failed: attrs.get('auto_pad', b'NOTSET') == b'NOTSET' and attrs.get('kernel_shape', [1, 1]) == [1, 1]")
        if not (attrs['group'] == 1 and attrs['strides'] == [1, 1] and (attrs['pads'] == [0] * 4) and (attrs['dilations'] == [1, 1])):
            raise ValueError("Graph contract check failed: attrs['group'] == 1 and attrs['strides'] == [1, 1] and (attrs['pads'] == [0] * 4) and (attrs['dilations'] == [1, 1])")
        uses = cons[n.output[0]]
        if not len(uses) == 2:
            raise ValueError('Graph contract check failed: len(uses) == 2')
        sig = next((x for x in uses if x.op_type == 'Sigmoid'))
        mul = next((x for x in uses if x.op_type == 'Mul'))
        if not (set(mul.input) == {n.output[0], sig.output[0]} and len(cons[sig.output[0]]) == 1):
            raise ValueError('Graph contract check failed: set(mul.input) == {n.output[0], sig.output[0]} and len(cons[sig.output[0]]) == 1')
        if not (list(sig.input) == list(n.output) and len(n.output) == 1 and (len(sig.output) == 1) and (len(mul.output) == 1)):
            raise ValueError('Graph contract check failed: list(sig.input) == list(n.output) and len(n.output) == 1 and (len(sig.output) == 1) and (len(mul.output) == 1)')
        if not not (set(n.output) | set(sig.output)) & graph_outputs:
            raise ValueError('Internal pointwise value is a graph output')
        replacements[n.name] = helper.make_node('PointwiseConvSilu', list(n.input), list(mul.output), name=n.name + '/FusedSilu', domain='jumc.pointwise')
        remove.update([sig.name, mul.name])
        records.append(n.name)
    if not len(records) == 2:
        raise ValueError('Graph contract check failed: len(records) == 2')
    nodes = [replacements.get(n.name, n) for n in m.graph.node if n.name not in remove]
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    m.opset_import.append(helper.make_opsetid('jumc.pointwise', 1))
    if not [w.SerializeToString() for w in old.graph.initializer] == [w.SerializeToString() for w in m.graph.initializer]:
        raise ValueError('Graph contract check failed: [w.SerializeToString() for w in old.graph.initializer] == [w.SerializeToString() for w in m.graph.initializer]')
    for op, count in [('DepthwiseConvSilu', 7), ('Attention', 9)]:
        a = [n.SerializeToString() for n in old.graph.node if n.op_type == op]
        b = [n.SerializeToString() for n in m.graph.node if n.op_type == op]
        if not (a == b and len(a) == count):
            raise ValueError('Graph contract check failed: a == b and len(a) == count')
    onnx.checker.check_model(m)
    dest = output
    onnx.save(m, str(dest))
    return dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), output=str(output), output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), fused_nodes=records, count=len(records), initializers_byte_identical=True, previous_fusions_unchanged=True, numerical_validation='not_run; requires ARM custom operator libraries')

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
    parser.add_argument('--source', required=True, type=Path, help='Attention- and depthwise-fused JuMCNet FP32 ONNX graph.')
    parser.add_argument('--output', required=True, type=Path, help='New pointwise-fused ONNX path.')
    parser.add_argument('--report', type=Path, help='New JSON path; defaults to OUTPUT with .json suffix.')
    args = parser.parse_args()
    source, output, report_path = output_paths(parser, args.source, args.output, args.report)
    report = fuse(source, output)
    write_json(report_path, report)
    print('FUSED', report['count'], output, flush=True)
if __name__ == '__main__':
    main()

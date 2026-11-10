"""Fuse the nine JuMCNet unmasked attention blocks without changing weights.

Requires numpy and onnx. Optional --probe-inputs also requires onnxruntime.
Inputs must follow the original fixed-shape JuMCNet export naming/graph contract.
"""
from pathlib import Path
import argparse
import hashlib
import json

def fuse(src, dst):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    src, dst = (Path(src), Path(dst))
    m = onnx.load(str(src))
    original_weights = {v.name: v.SerializeToString() for v in m.graph.initializer}
    vals = {v.name: numpy_helper.to_array(v) for v in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type == 'Constant':
            vals[n.output[0]] = numpy_helper.to_array(next((a.t for a in n.attribute if a.name == 'value')))
    nodes = list(m.graph.node)
    index = {n.name: i for i, n in enumerate(nodes)}
    if not len(index) == len(nodes):
        raise ValueError('Node names must be unique')
    replacements = {}
    removed = set()
    record = []
    prefixes = [n.name[:-len('qkv/MatMul')] for n in nodes if n.op_type == 'MatMul' and n.name.endswith('/attn/qkv/MatMul')]
    if not len(prefixes) == 9:
        raise ValueError('Graph contract check failed: len(prefixes) == 9')
    expected = {'qkv/MatMul': 'MatMul', 'qkv/Add': 'Add', 'Reshape': 'Reshape', 'Transpose': 'Transpose', 'Split': 'Split', 'Squeeze': 'Squeeze', 'Squeeze_1': 'Squeeze', 'Squeeze_2': 'Squeeze', 'Transpose_1': 'Transpose', 'Mul': 'Mul', 'Mul_1': 'Mul', 'MatMul': 'MatMul', 'Softmax': 'Softmax', 'MatMul_1': 'MatMul', 'Transpose_2': 'Transpose', 'Reshape_1': 'Reshape'}
    expected.update({'Shape': 'Shape', 'Slice': 'Slice', 'Cast': 'Cast', 'Sqrt': 'Sqrt', 'Div': 'Div', 'Cast_1': 'Cast', 'Sqrt_1': 'Sqrt', 'Sqrt_2': 'Sqrt'})

    def attrs(n):
        return {a.name: helper.get_attribute_value(a) for a in n.attribute}
    for pre in prefixes:
        ns = {k: nodes[index[pre + k]] for k in expected}
        if not all((ns[k].op_type == v for k, v in expected.items())):
            raise ValueError('Graph contract check failed: all((ns[k].op_type == v for k, v in expected.items()))')
        qkv = ns['qkv/MatMul']
        add = ns['qkv/Add']
        end = ns['Reshape_1']
        if not qkv.output[0] in add.input:
            raise ValueError('Graph contract check failed: qkv.output[0] in add.input')
        bias = next((v for v in add.input if v != qkv.output[0]))
        weight = qkv.input[1]
        b, s, three, h, d = map(int, vals[ns['Reshape'].input[1]])
        hidden = h * d
        if not (three == 3 and vals[weight].shape == (hidden, 3 * hidden) and (vals[bias].shape == (3 * hidden,))):
            raise ValueError('Graph contract check failed: three == 3 and vals[weight].shape == (hidden, 3 * hidden) and (vals[bias].shape == (3 * hidden,))')
        if not (vals[weight].dtype == np.float32 and vals[bias].dtype == np.float32):
            raise ValueError('Graph contract check failed: vals[weight].dtype == np.float32 and vals[bias].dtype == np.float32')
        if not attrs(ns['Transpose'])['perm'] == [2, 0, 3, 1, 4]:
            raise ValueError("Graph contract check failed: attrs(ns['Transpose'])['perm'] == [2, 0, 3, 1, 4]")
        if not attrs(ns['Transpose_1'])['perm'] == [0, 1, 3, 2]:
            raise ValueError("Graph contract check failed: attrs(ns['Transpose_1'])['perm'] == [0, 1, 3, 2]")
        if not attrs(ns['Transpose_2'])['perm'] == [0, 2, 1, 3]:
            raise ValueError("Graph contract check failed: attrs(ns['Transpose_2'])['perm'] == [0, 2, 1, 3]")
        if not (attrs(ns['Split'])['axis'] == 0 and vals[ns['Split'].input[1]].tolist() == [1, 1, 1]):
            raise ValueError("Graph contract check failed: attrs(ns['Split'])['axis'] == 0 and vals[ns['Split'].input[1]].tolist() == [1, 1, 1]")
        if not attrs(ns['Softmax'])['axis'] == -1:
            raise ValueError("Graph contract check failed: attrs(ns['Softmax'])['axis'] == -1")
        if not vals[end.input[1]].tolist() == [b, s, hidden]:
            raise ValueError('Graph contract check failed: vals[end.input[1]].tolist() == [b, s, hidden]')
        if not min(b, s, h, d) > 0:
            raise ValueError('Graph contract check failed: min(b, s, h, d) > 0')
        links = {'Reshape': ['qkv/Add'], 'Transpose': ['Reshape'], 'Split': ['Transpose'], 'Transpose_1': ['Squeeze_1'], 'Shape': ['Squeeze'], 'Slice': ['Shape'], 'Cast': ['Slice'], 'Sqrt': ['Cast'], 'Cast_1': ['Div'], 'Sqrt_1': ['Cast_1'], 'Sqrt_2': ['Cast_1'], 'MatMul': ['Mul', 'Mul_1'], 'Softmax': ['MatMul'], 'MatMul_1': ['Softmax', 'Squeeze_2'], 'Transpose_2': ['MatMul_1'], 'Reshape_1': ['Transpose_2']}
        for key, sources in links.items():
            if not list(ns[key].input[:len(sources)]) == [ns[source].output[0] for source in sources]:
                raise ValueError('Attention edge mismatch: ' + pre + key)
        if not len(ns['Split'].output) == 3:
            raise ValueError("Graph contract check failed: len(ns['Split'].output) == 3")
        for k, key in enumerate(('Squeeze', 'Squeeze_1', 'Squeeze_2')):
            if not ns[key].input[0] == ns['Split'].output[k]:
                raise ValueError("Graph contract check failed: ns[key].input[0] == ns['Split'].output[k]")
            if not vals[ns[key].input[1]].tolist() == [0]:
                raise ValueError('Graph contract check failed: vals[ns[key].input[1]].tolist() == [0]')
        if not set(ns['Mul'].input) == {ns['Squeeze'].output[0], ns['Sqrt_1'].output[0]}:
            raise ValueError("Graph contract check failed: set(ns['Mul'].input) == {ns['Squeeze'].output[0], ns['Sqrt_1'].output[0]}")
        if not set(ns['Mul_1'].input) == {ns['Transpose_1'].output[0], ns['Sqrt_2'].output[0]}:
            raise ValueError("Graph contract check failed: set(ns['Mul_1'].input) == {ns['Transpose_1'].output[0], ns['Sqrt_2'].output[0]}")
        if not len(ns['Slice'].input) == 3:
            raise ValueError("Graph contract check failed: len(ns['Slice'].input) == 3")
        if not vals[ns['Slice'].input[1]].tolist() == [-1]:
            raise ValueError("Graph contract check failed: vals[ns['Slice'].input[1]].tolist() == [-1]")
        if not vals[ns['Slice'].input[2]].tolist() == [np.iinfo(np.int64).max]:
            raise ValueError("Graph contract check failed: vals[ns['Slice'].input[2]].tolist() == [np.iinfo(np.int64).max]")
        if not attrs(ns['Shape']) == {}:
            raise ValueError("Graph contract check failed: attrs(ns['Shape']) == {}")
        if not attrs(ns['Cast']) == {'to': onnx.TensorProto.FLOAT}:
            raise ValueError("Graph contract check failed: attrs(ns['Cast']) == {'to': onnx.TensorProto.FLOAT}")
        if not attrs(ns['Cast_1']) == {'to': onnx.TensorProto.FLOAT}:
            raise ValueError("Graph contract check failed: attrs(ns['Cast_1']) == {'to': onnx.TensorProto.FLOAT}")
        if not ns['Div'].input[1] == ns['Sqrt'].output[0]:
            raise ValueError("Graph contract check failed: ns['Div'].input[1] == ns['Sqrt'].output[0]")
        if not (vals[ns['Div'].input[0]].shape == (1,) and vals[ns['Div'].input[0]].tolist() == [1.0]):
            raise ValueError("Graph contract check failed: vals[ns['Div'].input[0]].shape == (1,) and vals[ns['Div'].input[0]].tolist() == [1.0]")
        if not attrs(ns['Reshape']).get('allowzero', 0) == 0:
            raise ValueError("Graph contract check failed: attrs(ns['Reshape']).get('allowzero', 0) == 0")
        if not attrs(end).get('allowzero', 0) == 0:
            raise ValueError("Graph contract check failed: attrs(end).get('allowzero', 0) == 0")
        start_i = index[pre + 'qkv/MatMul']
        end_i = index[pre + 'Reshape_1']
        block = set(range(start_i, end_i + 1))
        block_outputs = {o for i in block for o in nodes[i].output}
        if not not ({v.name for v in m.graph.output} & block_outputs) - set(end.output):
            raise ValueError('Internal attention value is a graph output')
        if not all((nodes[i].name in {n.name for n in ns.values()} or nodes[i].op_type == 'Constant' for i in block)):
            raise ValueError('Unexpected node in attention block')
        for i, n in enumerate(nodes):
            if i not in block:
                if not not (set(n.input) & block_outputs) - set(end.output):
                    raise ValueError(n.name)
        node = helper.make_node('Attention', [qkv.input[0], weight, bias], list(end.output), name=pre + 'FusedAttention', domain='com.microsoft', num_heads=h, unidirectional=0, scale=float(1 / np.sqrt(d)))
        replacements[start_i] = node
        removed.update(block)
        record.append(dict(prefix=pre, batch=b, tokens=s, heads=h, head_dim=d, removed_nodes=len(block), scale=float(1 / np.sqrt(d))))
    new = []
    for i, n in enumerate(nodes):
        if i in replacements:
            new.append(replacements[i])
        elif i not in removed:
            new.append(n)
    required = {x.name for x in m.graph.output}
    kept = []
    for n in reversed(new):
        if required.intersection(n.output):
            kept.append(n)
            required.update(n.input)
    kept.reverse()
    del m.graph.node[:]
    m.graph.node.extend(kept)
    retained = [v for v in m.graph.initializer if v.name in required]
    del m.graph.initializer[:]
    m.graph.initializer.extend(retained)
    if not all((original_weights[v.name] == v.SerializeToString() for v in m.graph.initializer)):
        raise ValueError('Graph contract check failed: all((original_weights[v.name] == v.SerializeToString() for v in m.graph.initializer))')
    if not set(original_weights) == {v.name for v in m.graph.initializer}:
        raise ValueError('Unexpected removal of learned parameters')
    if not any((x.domain == 'com.microsoft' for x in m.opset_import)):
        m.opset_import.append(helper.make_opsetid('com.microsoft', 1))
    del m.graph.value_info[:]
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    return dict(before_nodes=len(nodes), after_nodes=len(kept), fused_attention_blocks=record, all_weight_tensors_byte_identical=True, sha256=hashlib.sha256(dst.read_bytes()).hexdigest(), bytes=dst.stat().st_size)

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

def probe(source, output, inputs_path, threads):
    import numpy as np
    import onnxruntime as ort
    loaded = np.load(inputs_path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            bank = loaded['inputs']
    else:
        bank = loaded
    if bank.dtype != np.float32 or bank.ndim != 4 or len(bank) == 0:
        raise ValueError('Probe inputs must be a nonempty FP32 NCHW array (NPY or NPZ key inputs).')
    outputs = []
    for path in (source, output):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        if len(session.get_inputs()) != 1:
            raise ValueError('The probe supports a single model input.')
        name = session.get_inputs()[0].name
        outputs.append(np.concatenate([session.run(None, {name: x[None]})[0] for x in bank]))
    y0, y1 = outputs
    return dict(n=len(bank), inputs=str(inputs_path.resolve()), inputs_sha256=hashlib.sha256(inputs_path.read_bytes()).hexdigest(), max_abs=float(np.abs(y0 - y1).max()), predictions_equal=bool(np.array_equal(y0.argmax(1), y1.argmax(1))), allclose=bool(np.allclose(y0, y1, atol=0.0002, rtol=0.0002)), atol=0.0002, rtol=0.0002)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Unfused JuMCNet FP32 ONNX graph.')
    parser.add_argument('--output', required=True, type=Path, help='New fused ONNX path.')
    parser.add_argument('--report', type=Path, help='New JSON path; defaults to OUTPUT with .json suffix.')
    parser.add_argument('--probe-inputs', type=Path, help='Optional NPY or NPZ inputs key, shaped NCHW.')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive.')
    if args.probe_inputs and (not args.probe_inputs.is_file()):
        parser.error('Probe inputs are missing: ' + str(args.probe_inputs))
    source, output, report_path = output_paths(parser, args.source, args.output, args.report)
    report = fuse(source, output)
    report.update(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), output=str(output), numerical_validation='not_run')
    if args.probe_inputs:
        report['probe_check'] = probe(source, output, args.probe_inputs, args.threads)
        passed = report['probe_check']['predictions_equal'] and report['probe_check']['allclose']
        report['numerical_validation'] = 'passed' if passed else 'failed'
        write_json(report_path, report)
        if not passed:
            raise RuntimeError('Numerical equivalence check failed; see ' + str(report_path))
    else:
        write_json(report_path, report)
    print('FUSED', len(report['fused_attention_blocks']), output, flush=True)
if __name__ == '__main__':
    main()

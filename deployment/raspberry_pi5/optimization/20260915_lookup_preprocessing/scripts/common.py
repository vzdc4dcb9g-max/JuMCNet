from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image

MEAN = np.array([.485, .456, .406], dtype=np.float32)[:, None, None]
STD = np.array([.229, .224, .225], dtype=np.float32)[:, None, None]

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def preprocess(path):
    # Exact PIL path used by torchvision Resize(256, BICUBIC), CenterCrop(224).
    with Image.open(path) as im:
        im = im.convert('RGB')
        w, h = im.size
        if w <= h:
            nw, nh = 256, int(256 * h / w)
        else:
            nw, nh = int(256 * w / h), 256
        if im.size != (nw, nh):
            im = im.resize((nw, nh), Image.Resampling.BICUBIC)
        left, top = int(round((nw - 224) / 2.)), int(round((nh - 224) / 2.))
        im = im.crop((left, top, left + 224, top + 224))
        a = np.array(im, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    return np.ascontiguousarray(((a - MEAN) / STD)[None], dtype=np.float32)

def metrics(target, pred, classes=6):
    cm = np.bincount(classes * np.asarray(target, dtype=int) + np.asarray(pred, dtype=int), minlength=classes**2).reshape(classes, classes)
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return dict(accuracy=float(tp.sum()/cm.sum()), macro_precision=float(precision.mean()), macro_recall=float(recall.mean()), macro_f1=float(f1.mean()), per_class_precision=precision.tolist(), per_class_recall=recall.tolist(), per_class_f1=f1.tolist(), per_class_support=cm.sum(1).tolist(), confusion_matrix=cm.tolist())

def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def session(path, threads=4):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])

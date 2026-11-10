"""Use the tested FP32 Attention-fused model with original preprocessing."""
from pathlib import Path
import argparse,json
from common import session,preprocess
from runners import Runner
import numpy as np

def main():
    p=argparse.ArgumentParser(description='JuMCNet FP32 inference, same inputs and classes as the benchmark')
    p.add_argument('--model',required=True,type=Path);p.add_argument('--image',required=True,type=Path)
    p.add_argument('--reuse-buffers',action='store_true',help='Optional method2; test results determine whether this is useful')
    a=p.parse_args();s=session(a.model,4);run=Runner(s,'buffer' if a.reuse_buffers else 'baseline')
    y=run(preprocess(a.image))[0];prob=np.exp(y-y.max());prob/=prob.sum();pred=int(y.argmax())
    print(json.dumps(dict(class_index=pred,class_name=['bianxing','gantiao','haozao','liekou','meibian','niaozhuo'][pred],confidence=float(prob[pred]),logits=y.tolist()),ensure_ascii=False))
if __name__=='__main__':main()

"""Method3 FP32 graph with exactly equivalent lookup preprocessing."""
from pathlib import Path
import argparse,json,numpy as np
from common import session
from preprocess_candidates import lookup

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--image',type=Path,required=True);a=p.parse_args()
    s=session(a.model,4);y=s.run(None,{'image':lookup(a.image)})[0][0]
    prob=np.exp(y-y.max());prob/=prob.sum();pred=int(y.argmax())
    print(json.dumps(dict(class_index=pred,class_name=['Deformed','Shriveled','Normal','Cracked','Moldy','Bird-pecked'][pred],confidence=float(prob[pred]),logits=y.tolist())))
if __name__=='__main__':main()

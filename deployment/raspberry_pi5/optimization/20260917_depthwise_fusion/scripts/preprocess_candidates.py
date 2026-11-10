"""Exact RGB/PIL geometry; optimize only the uint8-to-normalized-FP32 conversion."""
import numpy as np
from PIL import Image
from common import MEAN,STD,preprocess

LUT=((np.arange(256,dtype=np.float32)[None,:]/np.float32(255)-MEAN[:,0,0,None])/STD[:,0,0,None])

def crop_rgb(path):
    with Image.open(path) as im:
        im=im.convert('RGB');w,h=im.size
        nw,nh=(256,int(256*h/w)) if w<=h else (int(256*w/h),256)
        if im.size!=(nw,nh):im=im.resize((nw,nh),Image.Resampling.BICUBIC)
        left,top=int(round((nw-224)/2.)),int(round((nh-224)/2.))
        return im.crop((left,top,left+224,top+224))

def inplace(path):
    im=crop_rgb(path)
    out=np.empty((1,3,224,224),dtype=np.float32)
    out[0]=np.asarray(im,dtype=np.uint8).transpose(2,0,1)
    np.divide(out,np.float32(255),out=out)
    np.subtract(out,MEAN,out=out)
    np.divide(out,STD,out=out)
    return out

def lookup(path):
    im=crop_rgb(path);pixels=np.asarray(im,dtype=np.uint8)
    out=np.empty((1,3,224,224),dtype=np.float32)
    for c in range(3):np.take(LUT[c],pixels[:,:,c],out=out[0,c],mode='clip')
    return out

METHODS={'original':preprocess,'inplace':inplace,'lookup':lookup}

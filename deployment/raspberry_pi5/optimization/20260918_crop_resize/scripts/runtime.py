import os
os.environ['OMP_WAIT_POLICY']='PASSIVE';os.environ['KMP_BLOCKTIME']='0'
from pathlib import Path
import ctypes,numpy as np,onnxruntime as ort
from PIL import Image
from preprocess_candidates import crop_rgb,LUT
R=Path(__file__).resolve().parents[1]
def options():
 o=ort.SessionOptions();o.intra_op_num_threads=4;o.inter_op_num_threads=1;o.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;o.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
 o.add_session_config_entry('session.intra_op.allow_spinning','0');o.add_session_config_entry('session.inter_op.allow_spinning','0')
 for n in ['dw_silu','pw_silu','pw24_silu','stem_silu','more_ops','direct','dw8','fc']:o.register_custom_ops_library(str(R/'scripts'/f'{n}.so'))
 return o
def session(model):return ort.InferenceSession(str(model),sess_options=options(),providers=['CPUExecutionProvider'])
lut=ctypes.CDLL(str(R/'scripts/lut.so'));lut.lut_chw.argtypes=[ctypes.c_void_p]*3+[ctypes.c_size_t];lut.lut_chw.restype=None
resize=ctypes.CDLL(str(R/'scripts/cropresize.so'));resize.crop_resize.argtypes=[ctypes.c_void_p]+[ctypes.c_int]*6+[ctypes.c_void_p];resize.crop_resize.restype=ctypes.c_int
def resized_pixels(im):
 im=im.convert('RGB');w,h=im.size;nw,nh=(256,int(256*h/w)) if w<=h else (int(256*w/h),256)
 left,top=int(round((nw-224)/2.)),int(round((nh-224)/2.));pixels=np.asarray(im,dtype=np.uint8);out=np.empty((224,224,3),np.uint8)
 ret=resize.crop_resize(pixels.ctypes.data,w,h,nw,nh,left,top,out.ctypes.data)
 if ret:raise RuntimeError('crop_resize status '+str(ret))
 return out
def candidate_pixels(path):
 with Image.open(path) as im:return resized_pixels(im)
def native(path,mode='baseline'):
 pixels=candidate_pixels(path) if mode in ['resize','combined_resize'] else np.asarray(crop_rgb(path),dtype=np.uint8)
 x=np.empty((1,3,224,224),np.float32);lut.lut_chw(pixels.ctypes.data,LUT.ctypes.data,x.ctypes.data,224*224);return x

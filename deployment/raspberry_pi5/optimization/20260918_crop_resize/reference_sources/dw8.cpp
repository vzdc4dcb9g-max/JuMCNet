// FP32 3x3 depthwise convolution + SiLU in one output pass.
// Narrow contract: NCHW, multiplier 1, dilation 1, padding 1, stride 1 or 2.
// Retains the existing ORT pool (4 intra threads), no additional thread pool.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include <arm_neon.h>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <memory>
extern "C" float32x4_t _ZGVnN4v_expf(float32x4_t);
struct Kernel {
  const OrtApi* api; int stride;
  Kernel(const OrtApi* a,const OrtKernelInfo* info):api(a) {
    int64_t s; Ort::ThrowOnError(api->KernelInfoGetAttribute_int64(info,"stride",&s));
    stride=(int)s;if(stride!=1&&stride!=2)throw std::runtime_error("stride");
  }
  struct Work {const float *x,*w,*b;float* y;int C,H,W,OH,OW,s;};
  static void Channel(void* p,size_t nc){
    Work& z=*static_cast<Work*>(p);int c=nc%z.C;
    const float* in=z.x+nc*z.H*z.W;const float* w=z.w+c*9;
    float* out=z.y+nc*z.OH*z.OW;
    for(int oy=0;oy<z.OH;++oy){
      int iy=oy*z.s-1,ox=0;
      while(ox<z.OW){
        int ix=ox*z.s-1;
        if(z.s==1&&iy>=0&&iy+2<z.H&&ix>=0&&ox+7<z.OW&&ix+11<z.W){
          auto a=vdupq_n_f32(z.b[c]),b=a;
          for(int ky=0;ky<3;ky++){
            const float* q=in+(iy+ky)*z.W+ix;auto v0=vld1q_f32(q),v1=vld1q_f32(q+4),v2=vld1q_f32(q+8);
            a=vfmaq_n_f32(a,v0,w[ky*3]);b=vfmaq_n_f32(b,v1,w[ky*3]);
            a=vfmaq_n_f32(a,vextq_f32(v0,v1,1),w[ky*3+1]);b=vfmaq_n_f32(b,vextq_f32(v1,v2,1),w[ky*3+1]);
            a=vfmaq_n_f32(a,vextq_f32(v0,v1,2),w[ky*3+2]);b=vfmaq_n_f32(b,vextq_f32(v1,v2,2),w[ky*3+2]);
          }
          auto one=vdupq_n_f32(1.0f);
          vst1q_f32(out+oy*z.OW+ox,vdivq_f32(a,vaddq_f32(one,_ZGVnN4v_expf(vnegq_f32(a)))));
          vst1q_f32(out+oy*z.OW+ox+4,vdivq_f32(b,vaddq_f32(one,_ZGVnN4v_expf(vnegq_f32(b)))));ox+=8;continue;
        }
        if(iy>=0&&iy+2<z.H&&ix>=0&&ox+3<z.OW&&ix+3*z.s+2<z.W){
          float32x4_t a=vdupq_n_f32(z.b[c]);
          for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){
            const float* q=in+(iy+ky)*z.W+ix+kx;
            // Four stride-2 elements, without reading beyond the last required element.
            float32x4_t v;
            if(z.s==1)v=vld1q_f32(q);
            else if(ix+9<z.W)v=vld2q_f32(q).val[0];
            else{float32x4_t gathered={q[0],q[2],q[4],q[6]};v=gathered;}
            a=vfmaq_n_f32(a,v,w[ky*3+kx]);
          }
          float32x4_t den=vaddq_f32(vdupq_n_f32(1.0f),_ZGVnN4v_expf(vnegq_f32(a)));
          vst1q_f32(out+oy*z.OW+ox,vdivq_f32(a,den));ox+=4;
        }else{
          float a=z.b[c];
          for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){
            int yy=iy+ky,xx=ix+kx;
            if(yy>=0&&yy<z.H&&xx>=0&&xx<z.W)a=std::fma(in[yy*z.W+xx],w[ky*3+kx],a);
          }
          out[oy*z.OW+ox]=a/(1.0f+std::exp(-a));++ox;
        }
      }
    }
  }
  OrtStatusPtr ComputeV2(OrtKernelContext* context) noexcept {
    try{
      Ort::KernelContext ctx(context);auto X=ctx.GetInput(0);auto Wt=ctx.GetInput(1);auto B=ctx.GetInput(2);
      auto d=X.GetTensorTypeAndShapeInfo().GetShape();auto wd=Wt.GetTensorTypeAndShapeInfo().GetShape();
      if(d.size()!=4||wd.size()!=4||wd[0]!=d[1]||wd[1]!=1||wd[2]!=3||wd[3]!=3)throw std::runtime_error("invalid depthwise shape");
      std::vector<int64_t> shape={d[0],d[1],(d[2]+1)/stride,(d[3]+1)/stride};
      // floor((H + 2*pad - kernel)/stride)+1 = floor((H-1)/stride)+1.
      shape[2]=(d[2]-1)/stride+1;shape[3]=(d[3]-1)/stride+1;
      auto Y=ctx.GetOutput(0,shape);Work z{X.GetTensorData<float>(),Wt.GetTensorData<float>(),B.GetTensorData<float>(),Y.GetTensorMutableData<float>(),(int)d[1],(int)d[2],(int)d[3],(int)shape[2],(int)shape[3],stride};
      return api->KernelContext_ParallelFor(context,Channel,d[0]*d[1],4,&z);
    }catch(const std::exception& e){return api->CreateStatus(ORT_FAIL,e.what());}
  }
};
struct Op:Ort::CustomOpBase<Op,Kernel,true>{
  const char* GetName()const{return "DepthwiseConvSilu";}
  const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
  size_t GetInputTypeCount()const{return 3;}
  size_t GetOutputTypeCount()const{return 1;}
  ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
  OrtStatusPtr CreateKernelV2(const OrtApi& api,const OrtKernelInfo* info,void** out)const noexcept{
    try{*out=new Kernel(&api,info);return nullptr;}catch(const std::exception& e){return api.CreateStatus(ORT_FAIL,e.what());}
  }
};
// Static domain must outlive all sessions that use it.
static Op op;
static std::unique_ptr<Ort::CustomOpDomain> domain;
extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base){
  const OrtApi* api=base->GetApi(ORT_API_VERSION);Ort::InitApi(api);
  try{if(!domain){domain=std::make_unique<Ort::CustomOpDomain>("jumc.dw8");domain->Add(&op);}return api->AddCustomOpDomain(options,*domain);}
  catch(const std::exception& e){return api->CreateStatus(ORT_FAIL,e.what());}
}

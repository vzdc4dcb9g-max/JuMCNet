// FP32 RGB stem: [N,3,H,W] -> [N,16,ceil(H/2),ceil(W/2)].
// Fixed 3x3, stride 2, padding 1, dilation 1. Learned weights unchanged.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include <arm_neon.h>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
extern "C" float32x4_t _ZGVnN4v_expf(float32x4_t);
namespace {
struct Kernel {
 const OrtApi* api;
 Kernel(const OrtApi* a,const OrtKernelInfo*):api(a){}
 struct Work {const float *x,*w,*b;float *y;int PH,PW,OH,OW;};
 static float32x4_t silu(float32x4_t a){return vdivq_f32(a,vaddq_f32(vdupq_n_f32(1),_ZGVnN4v_expf(vnegq_f32(a))));}
 static void Tile(void* ptr,size_t idx){
  auto& z=*static_cast<Work*>(ptr);int row=idx%z.OH,oc=(idx/z.OH%4)*4,batch=idx/(4*z.OH),p=0;
  const float* x=z.x+batch*3*z.PH*z.PW;float* y=z.y+batch*16*z.OH*z.OW;
  for(;p+7<z.OW;p+=8){
   float32x4_t a0=vdupq_n_f32(0),a1=a0,a2=a0,a3=a0,b0=a0,b1=a0,b2=a0,b3=a0;
   for(int ic=0;ic<3;++ic)for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){
    const float* src=x+(ic*z.PH+2*row+ky)*z.PW+2*p+kx;
    float32x4_t v=vld2q_f32(src).val[0],u=vld2q_f32(src+8).val[0];
    int k=ic*9+ky*3+kx;
    #define MAC(c) a##c=vfmaq_n_f32(a##c,v,z.w[(oc+c)*27+k]);b##c=vfmaq_n_f32(b##c,u,z.w[(oc+c)*27+k]);
    MAC(0) MAC(1) MAC(2) MAC(3)
    #undef MAC
   }
   #define STORE(c) vst1q_f32(y+((oc+c)*z.OH+row)*z.OW+p,silu(vaddq_f32(a##c,vdupq_n_f32(z.b[oc+c]))));vst1q_f32(y+((oc+c)*z.OH+row)*z.OW+p+4,silu(vaddq_f32(b##c,vdupq_n_f32(z.b[oc+c]))));
   STORE(0) STORE(1) STORE(2) STORE(3)
   #undef STORE
  }
  for(;p<z.OW;++p)for(int c=oc;c<oc+4;++c){
   float a=0;for(int ic=0;ic<3;++ic)for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx)
    a=std::fma(x[(ic*z.PH+2*row+ky)*z.PW+2*p+kx],z.w[c*27+ic*9+ky*3+kx],a);
   a+=z.b[c];y[(c*z.OH+row)*z.OW+p]=a/(1+std::exp(-a));
  }
 }
 OrtStatusPtr ComputeV2(OrtKernelContext* context) noexcept{
  try{
   Ort::KernelContext ctx(context);auto X=ctx.GetInput(0),W=ctx.GetInput(1),B=ctx.GetInput(2);
   auto d=X.GetTensorTypeAndShapeInfo().GetShape();
   if(d.size()!=4||d[0]<1||d[1]!=3||d[2]<1||d[3]<1||W.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{16,3,3,3}||B.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{16})throw std::runtime_error("stem shape contract violated");
   int N=d[0],H=d[2],WW=d[3],PH=H+2,PW=WW+4,OH=(H+1)/2,OW=(WW+1)/2;
   std::vector<float> padded((size_t)N*3*PH*PW,0);const float* input=X.GetTensorData<float>();
   for(int b=0;b<N;++b)for(int c=0;c<3;++c)for(int h=0;h<H;++h)
    std::memcpy(padded.data()+((b*3+c)*PH+h+1)*PW+1,input+((b*3+c)*H+h)*WW,WW*sizeof(float));
   auto Y=ctx.GetOutput(0,std::vector<int64_t>{N,16,OH,OW});
   Work z{padded.data(),W.GetTensorData<float>(),B.GetTensorData<float>(),Y.GetTensorMutableData<float>(),PH,PW,OH,OW};
   return api->KernelContext_ParallelFor(context,Tile,(size_t)N*4*OH,4,&z);
  }catch(const std::exception& e){return api->CreateStatus(ORT_FAIL,e.what());}
 }
};
struct Op:Ort::CustomOpBase<Op,Kernel,true>{
 const char* GetName()const{return "StemConvSilu";}
 const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
 size_t GetInputTypeCount()const{return 3;}size_t GetOutputTypeCount()const{return 1;}
 ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 OrtStatusPtr CreateKernelV2(const OrtApi& api,const OrtKernelInfo* info,void** out)const noexcept{try{*out=new Kernel(&api,info);return nullptr;}catch(const std::exception& e){return api.CreateStatus(ORT_FAIL,e.what());}}
};
static Op op;static std::unique_ptr<Ort::CustomOpDomain> domain;
}
extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base){
 const OrtApi* api=base->GetApi(ORT_API_VERSION);Ort::InitApi(api);
 try{if(!domain){domain=std::make_unique<Ort::CustomOpDomain>("jumc.stem");domain->Add(&op);}return api->AddCustomOpDomain(options,*domain);}catch(const std::exception& e){return api->CreateStatus(ORT_FAIL,e.what());}
}

// Restricted FP32 pointwise Conv+SiLU: NCHW, C_in=16, C_out=32,
// kernel=1, stride=1, padding=0, group=1. Reuses ORT's intra-op pool.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include <arm_neon.h>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <memory>
extern "C" float32x4_t _ZGVnN4v_expf(float32x4_t);
namespace {
struct Kernel {
 const OrtApi* api;
 Kernel(const OrtApi* a,const OrtKernelInfo*):api(a){}
 struct Work {const float *x,*w,*b;float *y;int S;};
 static float32x4_t silu(float32x4_t x){return vdivq_f32(x,vaddq_f32(vdupq_n_f32(1),_ZGVnN4v_expf(vnegq_f32(x))));}
 static void Tile(void* ptr,size_t index){
  Work& z=*static_cast<Work*>(ptr);int batch=index/32,part=index%4,oc=(index%32)/4*4;
  int start=z.S*part/4,end=z.S*(part+1)/4,p=start;
  const float* x=z.x+batch*16*z.S;float* y=z.y+batch*32*z.S;
  for(;p+7<end;p+=8){
   float32x4_t a0=vdupq_n_f32(0),a1=a0,a2=a0,a3=a0,b0=a0,b1=a0,b2=a0,b3=a0;
   for(int ic=0;ic<16;++ic){
    float32x4_t v=vld1q_f32(x+ic*z.S+p),u=vld1q_f32(x+ic*z.S+p+4);
    a0=vfmaq_n_f32(a0,v,z.w[(oc+0)*16+ic]);b0=vfmaq_n_f32(b0,u,z.w[(oc+0)*16+ic]);
    a1=vfmaq_n_f32(a1,v,z.w[(oc+1)*16+ic]);b1=vfmaq_n_f32(b1,u,z.w[(oc+1)*16+ic]);
    a2=vfmaq_n_f32(a2,v,z.w[(oc+2)*16+ic]);b2=vfmaq_n_f32(b2,u,z.w[(oc+2)*16+ic]);
    a3=vfmaq_n_f32(a3,v,z.w[(oc+3)*16+ic]);b3=vfmaq_n_f32(b3,u,z.w[(oc+3)*16+ic]);
   }
   #define SAVE(k) vst1q_f32(y+(oc+k)*z.S+p,silu(vaddq_f32(a##k,vdupq_n_f32(z.b[oc+k])))); vst1q_f32(y+(oc+k)*z.S+p+4,silu(vaddq_f32(b##k,vdupq_n_f32(z.b[oc+k]))));
   SAVE(0) SAVE(1) SAVE(2) SAVE(3)
   #undef SAVE
  }
  for(;p<end;++p)for(int c=oc;c<oc+4;++c){float a=0;for(int ic=0;ic<16;++ic)a=std::fma(x[ic*z.S+p],z.w[c*16+ic],a);a+=z.b[c];y[c*z.S+p]=a/(1+std::exp(-a));}
 }
 OrtStatusPtr ComputeV2(OrtKernelContext* context) noexcept{
  try{
   Ort::KernelContext ctx(context);auto X=ctx.GetInput(0),Wt=ctx.GetInput(1),B=ctx.GetInput(2);
   auto d=X.GetTensorTypeAndShapeInfo().GetShape(),wd=Wt.GetTensorTypeAndShapeInfo().GetShape(),bd=B.GetTensorTypeAndShapeInfo().GetShape();
   if(d.size()!=4||d[0]<1||d[1]!=16||d[2]<1||d[3]<1||wd!=std::vector<int64_t>{32,16,1,1}||bd!=std::vector<int64_t>{32})throw std::runtime_error("pointwise shape contract violated");
   auto Y=ctx.GetOutput(0,std::vector<int64_t>{d[0],32,d[2],d[3]});
   Work z{X.GetTensorData<float>(),Wt.GetTensorData<float>(),B.GetTensorData<float>(),Y.GetTensorMutableData<float>(),(int)(d[2]*d[3])};
   return api->KernelContext_ParallelFor(context,Tile,d[0]*32,4,&z);
  }catch(const std::exception& e){return api->CreateStatus(ORT_FAIL,e.what());}
 }
};
struct Op:Ort::CustomOpBase<Op,Kernel,true>{
 const char* GetName()const{return "PointwiseConvSilu";}
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
 try{if(!domain){domain=std::make_unique<Ort::CustomOpDomain>("jumc.pointwise");domain->Add(&op);}return api->AddCustomOpDomain(options,*domain);}catch(const std::exception& e){return api->CreateStatus(ORT_FAIL,e.what());}
}

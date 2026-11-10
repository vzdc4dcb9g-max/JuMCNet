// Two front FFN expansions: FP32 [B,S,64] * [64,128] + bias, then SiLU.
// Pack constant weights once. Fused epilogue writes only the activated output.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include <arm_neon.h>
#include <vector>
#include <memory>
#include <stdexcept>
extern "C" float32x4_t _ZGVnN4v_expf(float32x4_t);
namespace {
struct Kernel {
 const OrtApi* api;std::vector<float> packed;
 Kernel(const OrtApi* a,const OrtKernelInfo* info):api(a){
  Ort::ConstKernelInfo k(info);int isconst=0;auto w=k.GetTensorConstantInput(1,&isconst);
  if(!isconst||w.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{64,128})throw std::runtime_error("FcSilu requires constant [64,128] weights");
  const float* src=w.GetTensorData<float>();packed.resize(64*128);
  for(int block=0;block<16;block++)for(int j=0;j<64;j++)for(int c=0;c<8;c++)packed[(block*64+j)*8+c]=src[j*128+block*8+c];
 }
 struct Work{const float *x,*w,*b;float* y;int M;};
 static float32x4_t silu(float32x4_t a){return vdivq_f32(a,vaddq_f32(vdupq_n_f32(1),_ZGVnN4v_expf(vnegq_f32(a))));}
 static void Tile(void* ptr,size_t id){
  auto& z=*static_cast<Work*>(ptr);int row=id*4;
  for(int block=0;block<16;block++){
   auto a0=vdupq_n_f32(0),a1=a0,a2=a0,a3=a0,b0=a0,b1=a0,b2=a0,b3=a0;
   const float* wp=z.w+block*64*8;
   for(int k=0;k<64;k++){
    auto w0=vld1q_f32(wp+k*8),w1=vld1q_f32(wp+k*8+4);
    #define FMA(r) if(row+r<z.M){float x=z.x[(row+r)*64+k];a##r=vfmaq_n_f32(a##r,w0,x);b##r=vfmaq_n_f32(b##r,w1,x);}
    FMA(0) FMA(1) FMA(2) FMA(3)
    #undef FMA
   }
   auto c0=vld1q_f32(z.b+block*8),c1=vld1q_f32(z.b+block*8+4);
   #define STORE(r) if(row+r<z.M){float* d=z.y+(row+r)*128+block*8;vst1q_f32(d,silu(vaddq_f32(a##r,c0)));vst1q_f32(d+4,silu(vaddq_f32(b##r,c1)));}
   STORE(0) STORE(1) STORE(2) STORE(3)
   #undef STORE
  }
 }
 OrtStatusPtr ComputeV2(OrtKernelContext* context) noexcept{
  try{Ort::KernelContext ctx(context);auto x=ctx.GetInput(0),w=ctx.GetInput(1),b=ctx.GetInput(2);auto d=x.GetTensorTypeAndShapeInfo().GetShape();
   if(d.size()!=3||d[0]<1||d[1]<1||d[2]!=64||w.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{64,128}||b.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{128})throw std::runtime_error("FcSilu invalid shape");
   int M=d[0]*d[1];d[2]=128;auto y=ctx.GetOutput(0,d);Work z{x.GetTensorData<float>(),packed.data(),b.GetTensorData<float>(),y.GetTensorMutableData<float>(),M};
   return api->KernelContext_ParallelFor(context,Tile,(M+3)/4,4,&z);
  }catch(const std::exception& e){return api->CreateStatus(ORT_FAIL,e.what());}
 }
};
struct Op:Ort::CustomOpBase<Op,Kernel,true>{
 const char* GetName()const{return "FcSilu";}const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
 size_t GetInputTypeCount()const{return 3;}size_t GetOutputTypeCount()const{return 1;}
 ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 OrtStatusPtr CreateKernelV2(const OrtApi& a,const OrtKernelInfo* i,void** out)const noexcept{try{*out=new Kernel(&a,i);return nullptr;}catch(const std::exception& e){return a.CreateStatus(ORT_FAIL,e.what());}}
};
static Op op;static std::unique_ptr<Ort::CustomOpDomain> domain;
}
extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base){const OrtApi* api=base->GetApi(ORT_API_VERSION);Ort::InitApi(api);try{if(!domain){domain=std::make_unique<Ort::CustomOpDomain>("jumc.fc");domain->Add(&op);}return api->AddCustomOpDomain(options,*domain);}catch(const std::exception& e){return api->CreateStatus(ORT_FAIL,e.what());}}

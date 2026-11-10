#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
extern "C" float32x4_t _ZGVnN4v_expf(float32x4_t);
namespace {
struct Layout {
 const OrtApi* api;int dir,H,W,C;
 Layout(const OrtApi* a,const OrtKernelInfo* info):api(a){Ort::ConstKernelInfo k(info);dir=k.GetAttribute<int64_t>("direction");H=k.GetAttribute<int64_t>("height");W=k.GetAttribute<int64_t>("width");C=k.GetAttribute<int64_t>("channels");if(H%2||W%2||H<2||W<2||C<1)throw std::runtime_error("Invalid patch geometry");}
 struct Work{const float* x;float* y;int H,W,C,S,dir;};
 static void Tile(void* p,size_t id){
  auto& z=*static_cast<Work*>(p);int halfw=z.W/2;
  if(z.dir==0){
   int patch=id/z.S,t=id%z.S,pos=((t/halfw)*2+patch/2)*z.W+(t%halfw)*2+patch%2;
   float* dst=z.y+(patch*z.S+t)*z.C;
   for(int c=0;c<z.C;++c)dst[c]=z.x[c*z.H*z.W+pos];
  }else{
   int c=id/z.H,h=id%z.H;float* dst=z.y+(c*z.H+h)*z.W;
   for(int w=0;w<z.W;w++){int patch=(h%2)*2+w%2,t=(h/2)*halfw+w/2;dst[w]=z.x[(patch*z.S+t)*z.C+c];}
  }
 }
 OrtStatusPtr ComputeV2(OrtKernelContext* context) noexcept {
  try{Ort::KernelContext ctx(context);auto x=ctx.GetInput(0);int S=H*W/4;auto input=dir==0?std::vector<int64_t>{1,C,H,W}:std::vector<int64_t>{4,S,C};auto shape=dir==0?std::vector<int64_t>{4,S,C}:std::vector<int64_t>{1,C,H,W};
   if(x.GetTensorTypeAndShapeInfo().GetShape()!=input)throw std::runtime_error("PatchLayout shape mismatch");auto y=ctx.GetOutput(0,shape);Work z{x.GetTensorData<float>(),y.GetTensorMutableData<float>(),H,W,C,S,dir};return api->KernelContext_ParallelFor(context,Tile,dir==0?4*S:C*H,4,&z);
  }catch(const std::exception& e){return api->CreateStatus(ORT_FAIL,e.what());}
 }
};
// FP32 unmasked self-attention, four heads of width 16. QKV and learned bias
// are unchanged. Compute four queries per tile, immediately consume softmax
// probabilities, and avoid the full [batch,heads,tokens,tokens] matrix.
struct Attention {
 const OrtApi* api;Attention(const OrtApi* a,const OrtKernelInfo*):api(a){}
 struct Work{const float *x,*w,*b;float *qkv,*kt,*v,*y;int B,S,P,T;};
 static void StoreK(float* p,int P,float32x4_t a,float32x4_t b,float32x4_t c,float32x4_t d){
  auto ab=vtrnq_f32(a,b),cd=vtrnq_f32(c,d);
  vst1q_f32(p,vcombine_f32(vget_low_f32(ab.val[0]),vget_low_f32(cd.val[0])));
  vst1q_f32(p+P,vcombine_f32(vget_low_f32(ab.val[1]),vget_low_f32(cd.val[1])));
  vst1q_f32(p+2*P,vcombine_f32(vget_high_f32(ab.val[0]),vget_high_f32(cd.val[0])));
  vst1q_f32(p+3*P,vcombine_f32(vget_high_f32(ab.val[1]),vget_high_f32(cd.val[1])));
 }
 static void Project(void* p,size_t id){
  auto& z=*static_cast<Work*>(p);int batch=id/z.T,r=(id%z.T)*4;
  for(int col=0;col<192;col+=8){
   float32x4_t a0=vdupq_n_f32(0),a1=a0,a2=a0,a3=a0,b0=a0,b1=a0,b2=a0,b3=a0;
   for(int k=0;k<64;k++){
    auto w0=vld1q_f32(z.w+k*192+col),w1=vld1q_f32(z.w+k*192+col+4);
    #define PROJECT(i) if(r+i<z.S){float q=z.x[((batch*z.S+r+i)*64)+k];a##i=vfmaq_n_f32(a##i,w0,q);b##i=vfmaq_n_f32(b##i,w1,q);}
    PROJECT(0) PROJECT(1) PROJECT(2) PROJECT(3)
    #undef PROJECT
   }
   auto bias0=vld1q_f32(z.b+col),bias1=vld1q_f32(z.b+col+4);
   a0=vaddq_f32(a0,bias0);a1=vaddq_f32(a1,bias0);a2=vaddq_f32(a2,bias0);a3=vaddq_f32(a3,bias0);
   b0=vaddq_f32(b0,bias1);b1=vaddq_f32(b1,bias1);b2=vaddq_f32(b2,bias1);b3=vaddq_f32(b3,bias1);
   if(col<64){
    #define QS(i) if(r+i<z.S){float* d=z.qkv+(batch*z.S+r+i)*64+col;vst1q_f32(d,a##i);vst1q_f32(d+4,b##i);}
    QS(0) QS(1) QS(2) QS(3)
    #undef QS
   }else if(col<128){
    int head=(col-64)/16,dc=(col-64)%16;float* d=z.kt+((batch*4+head)*16+dc)*z.P+r;
    StoreK(d,z.P,a0,a1,a2,a3);StoreK(d+4*z.P,z.P,b0,b1,b2,b3);
   }else{
    int head=(col-128)/16,dc=(col-128)%16;
    #define VS(i) if(r+i<z.S){float* d=z.v+((batch*4+head)*z.P+r+i)*16+dc;vst1q_f32(d,a##i);vst1q_f32(d+4,b##i);}
    VS(0) VS(1) VS(2) VS(3)
    #undef VS
   }
  }
 }
 static void Attend(void* p,size_t id){
  auto& z=*static_cast<Work*>(p);int bh=id/z.T,q=(id%z.T)*4,batch=bh/4,head=bh%4;
  thread_local std::vector<float> score; if(score.size()<size_t(4*z.P))score.resize(4*z.P);
  float Q[4][16]{};
  for(int r=0;r<4;r++)if(q+r<z.S)for(int d=0;d<16;d++)Q[r][d]=z.qkv[(batch*z.S+q+r)*64+head*16+d];
  for(int k=0;k<z.P;k+=4){
   float32x4_t a0=vdupq_n_f32(0),a1=a0,a2=a0,a3=a0;
   for(int d=0;d<16;d++){auto v=vld1q_f32(z.kt+(bh*16+d)*z.P+k);a0=vfmaq_n_f32(a0,v,Q[0][d]);a1=vfmaq_n_f32(a1,v,Q[1][d]);a2=vfmaq_n_f32(a2,v,Q[2][d]);a3=vfmaq_n_f32(a3,v,Q[3][d]);}
   vst1q_f32(score.data()+k,vmulq_n_f32(a0,.25f));vst1q_f32(score.data()+z.P+k,vmulq_n_f32(a1,.25f));vst1q_f32(score.data()+2*z.P+k,vmulq_n_f32(a2,.25f));vst1q_f32(score.data()+3*z.P+k,vmulq_n_f32(a3,.25f));
  }
  for(int r=0;r<4;r++){
   float* s=score.data()+r*z.P;for(int k=z.S;k<z.P;k++)s[k]=-std::numeric_limits<float>::infinity();
   auto mx=vdupq_n_f32(-std::numeric_limits<float>::infinity());for(int k=0;k<z.P;k+=4)mx=vmaxq_f32(mx,vld1q_f32(s+k));mx=vdupq_n_f32(vmaxvq_f32(mx));auto sum=vdupq_n_f32(0);
   for(int k=0;k<z.P;k+=4){auto e=_ZGVnN4v_expf(vsubq_f32(vld1q_f32(s+k),mx));vst1q_f32(s+k,e);sum=vaddq_f32(sum,e);}float inv=1.f/vaddvq_f32(sum);for(int k=0;k<z.P;k+=4)vst1q_f32(s+k,vmulq_n_f32(vld1q_f32(s+k),inv));
  }
  float32x4_t a00=vdupq_n_f32(0),a01=a00,a02=a00,a03=a00,a10=a00,a11=a00,a12=a00,a13=a00,a20=a00,a21=a00,a22=a00,a23=a00,a30=a00,a31=a00,a32=a00,a33=a00;
  for(int k=0;k<z.S;k++){
   const float* v=z.v+(bh*z.P+k)*16;auto v0=vld1q_f32(v),v1=vld1q_f32(v+4),v2=vld1q_f32(v+8),v3=vld1q_f32(v+12);
   #define PV(r) {float p=score[r*z.P+k];a##r##0=vfmaq_n_f32(a##r##0,v0,p);a##r##1=vfmaq_n_f32(a##r##1,v1,p);a##r##2=vfmaq_n_f32(a##r##2,v2,p);a##r##3=vfmaq_n_f32(a##r##3,v3,p);}
   PV(0) PV(1) PV(2) PV(3)
   #undef PV
  }
  #define OUT(r) if(q+r<z.S){float* d=z.y+(batch*z.S+q+r)*64+head*16;vst1q_f32(d,a##r##0);vst1q_f32(d+4,a##r##1);vst1q_f32(d+8,a##r##2);vst1q_f32(d+12,a##r##3);}
  OUT(0) OUT(1) OUT(2) OUT(3)
  #undef OUT
 }
 OrtStatusPtr ComputeV2(OrtKernelContext* context) noexcept{
  try{Ort::KernelContext ctx(context);auto x=ctx.GetInput(0),w=ctx.GetInput(1),b=ctx.GetInput(2);auto dims=x.GetTensorTypeAndShapeInfo().GetShape();
   if(dims.size()!=3||dims[2]!=64||dims[0]<1||dims[1]<1||w.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{64,192}||b.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{192})throw std::runtime_error("TiledAttention shape mismatch");
   int B=dims[0],S=dims[1],P=(S+3)/4*4,T=(S+3)/4;auto y=ctx.GetOutput(0,dims);std::unique_ptr<float[]> qkv(new float[(size_t)B*S*64]),v(new float[(size_t)B*4*P*16]);std::vector<float> kt((size_t)B*4*16*P,0);Work z{x.GetTensorData<float>(),w.GetTensorData<float>(),b.GetTensorData<float>(),qkv.get(),kt.data(),v.get(),y.GetTensorMutableData<float>(),B,S,P,T};
   if(auto s=api->KernelContext_ParallelFor(context,Project,B*T,4,&z))return s;
   return api->KernelContext_ParallelFor(context,Attend,B*4*T,4,&z);
  }catch(const std::exception& e){return api->CreateStatus(ORT_FAIL,e.what());}
 }
};
template<class K,int Inputs>struct Op:Ort::CustomOpBase<Op<K,Inputs>,K,true>{
 const char* name;explicit Op(const char* n):name(n){}const char* GetName()const{return name;}const char* GetExecutionProviderType()const{return "CPUExecutionProvider";}
 size_t GetInputTypeCount()const{return Inputs;}size_t GetOutputTypeCount()const{return 1;}ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 OrtStatusPtr CreateKernelV2(const OrtApi& api,const OrtKernelInfo* info,void** out)const noexcept{try{*out=new K(&api,info);return nullptr;}catch(const std::exception& e){return api.CreateStatus(ORT_FAIL,e.what());}}
};
static Op<Layout,1> layout("PatchLayout");static Op<Attention,3> attn("TiledAttention");static std::unique_ptr<Ort::CustomOpDomain> domain;
}
extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base){const OrtApi* api=base->GetApi(ORT_API_VERSION);Ort::InitApi(api);try{if(!domain){domain=std::make_unique<Ort::CustomOpDomain>("jumc.direct");domain->Add(&layout);domain->Add(&attn);}return api->AddCustomOpDomain(options,*domain);}catch(const std::exception& e){return api->CreateStatus(ORT_FAIL,e.what());}}

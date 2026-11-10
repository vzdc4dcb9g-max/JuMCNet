// Crop-aware separable bicubic resampling, matching Pillow 12.2's RGB 8-bit
// algorithm (a=-0.5, normalized signed 22-bit coefficients, rounding per pass).
// Only the final 224x224 output region is computed; the original resize scale
// and sampling coordinates are unchanged. No approximate pixel arithmetic.
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <arm_neon.h>
namespace {
double cubic(double x){x=std::abs(x);if(x<1)return ((1.5*x-2.5)*x)*x+1;if(x<2)return (((x-5)*x+8)*x-4)*(-.5);return 0;}
struct Coeff{int first;std::vector<int> k;};
std::vector<Coeff> coeff(int in,int out,int offset){
 std::vector<Coeff> result(224);double scale=double(in)/out,fs=std::max(scale,1.),support=2*fs;
 for(int i=0;i<224;i++){double center=(offset+i+.5)*scale;int lo=std::max(0,int(center-support+.5)),hi=std::min(in,int(center+support+.5));auto& c=result[i];c.first=lo;std::vector<double> f(hi-lo);double sum=0;
  for(int j=0;j<hi-lo;j++){f[j]=cubic((j+lo-center+.5)*(1/fs));sum+=f[j];}
  for(auto v:f){if(sum!=0)v/=sum;c.k.push_back(int((v<0?-.5:.5)+v*(1<<22)));}
 }return result;
}
uint8_t clip(int x){return uint8_t(std::max(0,std::min(255,x>>22)));}
}
extern "C" int crop_resize(const uint8_t* src,int W,int H,int NW,int NH,int left,int top,uint8_t* dst){
 try{
  if(W<1||H<1||NW<224||NH<224||left<0||top<0||left+224>NW||top+224>NH)return 1;
  auto cx=coeff(W,NW,left),cy=coeff(H,NH,top);int first=cy.front().first,last=cy.back().first+cy.back().k.size();std::vector<uint8_t> tmp((last-first)*224*3);
  for(int y=first;y<last;y++)for(int x=0;x<224;x++){const auto& c=cx[x];int a=1<<21,b=a,d=a;
   for(size_t j=0;j<c.k.size();j++){const uint8_t* p=src+((size_t)y*W+c.first+j)*3;a+=p[0]*c.k[j];b+=p[1]*c.k[j];d+=p[2]*c.k[j];}
   uint8_t* o=tmp.data()+((y-first)*224+x)*3;o[0]=clip(a);o[1]=clip(b);o[2]=clip(d);
  }
  for(int y=0;y<224;y++){const auto& c=cy[y];int x=0;
   for(;x+3<224*3;x+=4){auto sum=vdupq_n_s32(1<<21);
    for(size_t j=0;j<c.k.size();j++){const uint8_t* p=tmp.data()+((c.first-first+j)*224*3+x);int32_t v[4]={p[0],p[1],p[2],p[3]};sum=vmlaq_n_s32(sum,vld1q_s32(v),c.k[j]);}
    sum=vminq_s32(vmaxq_s32(vshrq_n_s32(sum,22),vdupq_n_s32(0)),vdupq_n_s32(255));int32_t a[4];vst1q_s32(a,sum);for(int k=0;k<4;k++)dst[y*224*3+x+k]=uint8_t(a[k]);
   }
  }return 0;
 }catch(...){return 2;}
}

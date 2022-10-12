#include <random>
#include <cstdio>
#include <cmath>
#include <cstring>

#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"
#include "m1cycles.h"
#include "linux-perf-events.h"

typedef uint16_t bfloat16;

// al 8
// ax 16
// eax 32
// rax 64

struct AVXBF16 : Xbyak::CodeGenerator {
  AVXBF16() {
    align(16);
    Xbyak::util::StackFrame sf(this, 4);

    mov(ax, ptr[sf.p[0]]); // a : bf16 : 1 word
    vpbroadcastw(zm0, ax);
    
    vmovdqu16(zm1, ptr[sf.p[1]]); // b : bf16 : 32 words
    vmovdqu16(zm2, ptr[sf.p[2]]); // b : bf16 : 32 words

    vmovups(zm3, ptr[sf.p[3]   ]);   // c : fp32 : 16 words
    vmovups(zm4, ptr[sf.p[3]+64]);   // c : fp32 : 16 words
 
    vdpbf16ps(zm3, zm1, zm0); // c += a*b
    vdpbf16ps(zm4, zm2, zm0); // c += a*b

    vmovups(ptr[sf.p[3]   ], zm3); // c : fp32 : 16 wrods
    vmovups(ptr[sf.p[3]+64], zm4); // c : fp32 : 16 wrods
  }
};

struct AVXBF16_VER2 : Xbyak::CodeGenerator {
  AVXBF16_VER2() {
    align(16);
    Xbyak::util::StackFrame sf(this, 4);

    mov(eax, ptr[sf.p[0]]); // a : bf16 : 2 words
    vpbroadcastd(zm0, eax);

    vmovdqu16(zm1, ptr[sf.p[1]]); // b : bf16 : 32 words
    vmovdqu16(zm2, ptr[sf.p[2]]); // b : bf16 : 32 words

    vmovups(zm3, ptr[sf.p[3]   ]);   // c : fp32 : 16 words
    vmovups(zm4, ptr[sf.p[3]+64]);   // c : fp32 : 16 words
 
    vdpbf16ps(zm3, zm1, zm0); // c += a*b
    vdpbf16ps(zm4, zm2, zm0); // c += a*b

    vmovups(ptr[sf.p[3]   ], zm3); // c : fp32 : 16 wrods
    vmovups(ptr[sf.p[3]+64], zm4); // c : fp32 : 16 wrods
  }
};

struct AVXBF16_VER3 : Xbyak::CodeGenerator {
  AVXBF16_VER3() {
    align(16);
    Xbyak::util::StackFrame sf(this, 3);

    mov(eax, ptr[sf.p[0]]); // a : bf16 : 2 words
    vpbroadcastd(zm0, eax);

    vmovdqu16(zm1, ptr[sf.p[1]   ]); // b : bf16 : 32 words
    vmovdqu16(zm2, ptr[sf.p[1]+64]); // b : bf16 : 32 words

    vmovups(zm3, ptr[sf.p[2]   ]);   // c : fp32 : 16 words
    vmovups(zm4, ptr[sf.p[2]+64]);   // c : fp32 : 16 words
 
    vdpbf16ps(zm3, zm1, zm0); // c += a*b
    vdpbf16ps(zm4, zm2, zm0); // c += a*b

    vmovups(ptr[sf.p[2]   ], zm3); // c : fp32 : 16 wrods
    vmovups(ptr[sf.p[2]+64], zm4); // c : fp32 : 16 wrods
  }
};


struct AVXBF16_VER4 : Xbyak::CodeGenerator {
  AVXBF16_VER4() {
    align(16);
    Xbyak::util::StackFrame sf(this, 3);
    vmovups(zm3, ptr[sf.p[2]   ]);   // c : fp32 : 16 words
    vmovups(zm4, ptr[sf.p[2]+64]);   // c : fp32 : 16 words

    mov(eax, ptr[sf.p[0]]  ); // a : bf16 : 2 words
    vpbroadcastd(zm0, eax);

    mov(eax, ptr[sf.p[0]+4]); // a : bf16 : 2 words
    vpbroadcastd(zm5, eax);

    vmovdqu16(zm1, ptr[sf.p[1]   ]); // b : bf16 : 32 words
    vmovdqu16(zm2, ptr[sf.p[1]+64]); // b : bf16 : 32 words

    vmovdqu16(zm6, ptr[sf.p[1]+128]); // b : bf16 : 32 words
    vmovdqu16(zm7, ptr[sf.p[1]+192]); // b : bf16 : 32 words
 
    vdpbf16ps(zm3, zm1, zm0); // c += a*b
    vdpbf16ps(zm4, zm2, zm0); // c += a*b
    vdpbf16ps(zm3, zm6, zm5); // c += a*b
    vdpbf16ps(zm4, zm7, zm5); // c += a*b
    
    vmovups(ptr[sf.p[2]   ], zm3); // c : fp32 : 16 wrods
    vmovups(ptr[sf.p[2]+64], zm4); // c : fp32 : 16 wrods
  }
};

union _d2i {
  double x;
  uint64_t v;
};

union _f2i {
  float x;
  uint32_t v;
};
  
uint64_t d2i(double x) {
  union _d2i now;
  now.x = x;
  return now.v;
}

uint32_t f2i(float x) {
  union _f2i now;
  now.x = x;
  return now.v;
}

bfloat16 f2b(float x) {
  uint32_t v = f2i(x);
  bfloat16 t = v >> 16; // no rounding 
  return t;
}
  
float b2f(bfloat16 x) {
  union _f2i t;
  t.v = x << 16;
  return t.x;
}

void mm_ref(bfloat16 *a, bfloat16 *b, float *c, const int n, const int lda)
{
  // reference A^N x B^N
  for(int j = 0; j < n; j++) {
    for(int i = 0; i < n; i++) {
      float tmp = 0.0;
      for(int k = 0; k < n; k++) {
	tmp += b2f(a[j*lda+k]) * b2f(b[k*lda+i]);
      }
      c[j*lda+i] = tmp;
    }
  }
}

void mm_outer(bfloat16 *a, bfloat16 *b, float *c, const int n, const int lda)
{
  for(int k = 0; k < n; k++) {
    bfloat16 at[n], bn[n];
    for(int i = 0; i < n; i++) {
      at[i] = a[i*lda + k];
      bn[i] = b[k*lda + i];
    }
    for(int j = 0; j < n; j++) {
      float atmp = b2f(at[j]);
      for(int i = 0; i < n; i++) {
	c[j*lda+i] += atmp * b2f(bn[i]);	
      }
    }
  }
}

void mm_outer_avx_ver1(bfloat16 *a, bfloat16 *b, float *c, const int n, const int lda)
{
  AVXBF16 code;
  auto f = code.getCode<void (*)(const void *, const void *, const void *, void *)>();    

  for(int k = 0; k < n; k++) {
    bfloat16 at[n], bn[n];
    for(int i = 0; i < n; i++) {
      at[i] = a[i*lda + k];
      bn[i] = b[k*lda + i];
    }

    bfloat16 btmp1[32];
    bfloat16 btmp2[32];

    for(int p = 0; p < 16; p++) {
      btmp1[2*p  ] = bn[p];
      btmp1[2*p+1] = 0x0;
      btmp2[2*p]   = bn[p+16];
      btmp2[2*p+1] = 0x0;
    }

    for(int j = 0; j < n; j++) {
      f(&at[j], btmp1, btmp2, &c[j*lda+  0]);
    }
  }
}

void mm_outer_avx_ver2(bfloat16 *a, bfloat16 *b, float *c, const int n, const int lda)
{
  AVXBF16_VER2 code;
  auto f = code.getCode<void (*)(const void *, const void *, const void *, void *)>();    
  
  for(int k = 0; k < n; k += 2) {
    bfloat16 at1[n], bn1[n];
    bfloat16 at2[n], bn2[n];

    int k1 = k;
    int k2 = k+1;
      
    for(int i = 0; i < n; i++) {
      at1[i] = a[i*lda + k1];
      bn1[i] = b[k1*lda + i];
      at2[i] = a[i*lda + k2];
      bn2[i] = b[k2*lda + i];
    }
    bfloat16 btmp1[32];
    bfloat16 btmp2[32];

    for(int p = 0; p < 16; p++) {
      btmp1[2*p  ] = bn1[p];
      btmp1[2*p+1] = bn2[p];
      btmp2[2*p  ] = bn1[p+16];
      btmp2[2*p+1] = bn2[p+16];
    }

    for(int j = 0; j < n; j++) {
      bfloat16 atmp[2] = {at1[j], at2[j]};
      f(&atmp, btmp1, btmp2, &c[j*lda+  0]);
    }
  }
}

void mm_outer_avx_ver3(bfloat16 *a, bfloat16 *b, float *c, const int n, const int lda)
{
  AVXBF16_VER3 code;
  auto f = code.getCode<void (*)(const void *, const void *, void *)>();    
  
  for(int k = 0; k < n; k += 4) {
    bfloat16 at1[n], bn1[n];
    bfloat16 at2[n], bn2[n];
    bfloat16 at3[n], bn3[n];
    bfloat16 at4[n], bn4[n];

    int k1 = k;
    int k2 = k+1;
    int k3 = k+2;
    int k4 = k+3;
      
    for(int i = 0; i < n; i++) {
      at1[i] = a[i*lda + k1];
      bn1[i] = b[k1*lda + i];
      at2[i] = a[i*lda + k2];
      bn2[i] = b[k2*lda + i];
      at3[i] = a[i*lda + k3];
      bn3[i] = b[k3*lda + i];
      at4[i] = a[i*lda + k4];
      bn4[i] = b[k4*lda + i];
    }

    bfloat16 btmp1[32*2];
    bfloat16 btmp2[32*2];
    for(int p = 0; p < 32; p++) {
      btmp1[2*p  ] = bn1[p];
      btmp1[2*p+1] = bn2[p];
      btmp2[2*p  ] = bn3[p];
      btmp2[2*p+1] = bn4[p];
    }
    
    for(int j = 0; j < n; j++) {
      bfloat16 atmp1[2] = {at1[j], at2[j]};
      bfloat16 atmp2[2] = {at3[j], at4[j]};
      f(&atmp1, btmp1, &c[j*lda+0]);
      f(&atmp2, btmp2, &c[j*lda+0]);
    }
  }
}

void mm_outer_avx_ver4(bfloat16 *a, bfloat16 *b, float *c, const int n, const int lda)
{
  AVXBF16_VER4 code;
  auto f = code.getCode<void (*)(const void *, const void *, void *)>();    
  
  for(int k = 0; k < n; k += 4) {
    bfloat16 btmpx[32*2*2];

    int k0 = k;
    int k1 = k+1;
    int k2 = k+2;
    int k3 = k+3;
    for(int p = 0; p < 32; p++) {
      btmpx[2*p     ] = b[k0*lda + p]; 
      btmpx[2*p+1   ] = b[k1*lda + p]; 
      btmpx[2*p  +64] = b[k2*lda + p]; 
      btmpx[2*p+1+64] = b[k3*lda + p]; 
    }

    for(int j = 0; j < n; j++) {
      f(&a[j*lda+k], btmpx, &c[j*lda+0]);
    }
  }
}

performance_counters measure(const int LOOP, void (*func)(bfloat16 *, bfloat16 *, float *, const int, const int),
			     bfloat16 *a, bfloat16 *b, float *c, const int n, const int lda)
{
  performance_counters start(0.0), end(0.0), diff(0.0);
  for(int l = 0; l < LOOP; l++) {

    for(int i = 0; i < n*n; i++) c[i] = 0.0;

    start = get_counters();
    func(a, b, c, n, lda);
    end = get_counters(true);
    diff += end - start;
  }
  diff /= (double)LOOP;
  
  return diff;
}

double L1_norm(float *ref, float *res, const int n)
{
  double sum = 0.0;
  for(int i = 0; i < n*n; i++) {
    sum += fabsf(ref[i] - res[i]);
  }
  return sum;
}

void output(performance_counters &now, float *ref, float *res, double ops)
{
  std::cout << now.cycles << "\t" << now.instructions << "\t"
	    << round(ops/now.cycles * 100.0)/100.0 << "\t"
	    << L1_norm(ref, res, 32) << "\n";
}

int main()
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<> urdist(0.0, 1.0);

  setup_performance_counters();

  try {
    bfloat16 *a, *b; // input
    float *ref, *res1, *res2, *res3, *res4, *res5;
      
    a = new bfloat16[32*32];
    b = new bfloat16[32*32];

    ref  = new float[32*32];
    res1 = new float[32*32];
    res2 = new float[32*32];
    res3 = new float[32*32];
    res4 = new float[32*32];
    res5 = new float[32*32];

    for(int j = 0; j < 32; j++) {
      for(int i = 0; i < 32; i++) {
	a[j*32+i] = f2b(urdist(mt));
	b[j*32+i] = f2b(urdist(mt));
      }
    }

    double ops = 2.0*pow(32.0, 3.0);
    
    const int LOOP = 225;
    performance_counters now;

    now = measure(LOOP, mm_ref, a, b, ref, 32, 32);
    std::cout << "ref            "; output(now, ref, ref, ops);

    now = measure(LOOP, mm_outer, a, b, res1, 32, 32);
    std::cout << "outer          "; output(now, ref, res1, ops);

    now = measure(LOOP, mm_outer_avx_ver1, a, b, res2, 32, 32);
    std::cout << "outer avx ver1 "; output(now, ref, res2, ops);

    now = measure(LOOP, mm_outer_avx_ver2, a, b, res3, 32, 32);
    std::cout << "outer avx ver2 "; output(now, ref, res3, ops);

    now = measure(LOOP, mm_outer_avx_ver3, a, b, res4, 32, 32);
    std::cout << "outer avx ver3 "; output(now, ref, res4, ops);

    now = measure(LOOP, mm_outer_avx_ver4, a, b, res5, 32, 32);
    std::cout << "outer avx ver4 "; output(now, ref, res5, ops);

  } catch (std::exception& e) {
    printf("ERR:%s\n", e.what());
  } catch (...) {
    printf("unknown error\n");
  }
}


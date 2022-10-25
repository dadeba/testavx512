#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

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
    vdpbf16ps(zm3, zm6, zm5); // c += a*b

    vdpbf16ps(zm4, zm2, zm0); // c += a*b
    vdpbf16ps(zm4, zm7, zm5); // c += a*b
    
    vmovups(ptr[sf.p[2]   ], zm3); // c : fp32 : 16 wrods
    vmovups(ptr[sf.p[2]+64], zm4); // c : fp32 : 16 wrods
  }
};


struct AVXBF16_VER4x : Xbyak::CodeGenerator {
  const unsigned int off = 64;
  
  AVXBF16_VER4x() {
    align(16);
    Xbyak::util::StackFrame sf(this, 4);
    vmovdqu16(zm1, ptr[sf.p[1]   ]); // b : bf16 : 32 words
    vmovdqu16(zm2, ptr[sf.p[1]+64]); // b : bf16 : 32 words
    vmovdqu16(zm6, ptr[sf.p[1]+128]); // b : bf16 : 32 words
    vmovdqu16(zm7, ptr[sf.p[1]+192]); // b : bf16 : 32 words

    xor_(r9, r9); // j
L(".loop");
    mov(r10, r9);
    shl(r10, 5);   // j*lda (j<<5)

    mov(r11, r10);
    shl(r11, 2);                // 32bit
    add(r11, sf.p[2]);          // &c[j*lda];
    vmovups(zm3, ptr[r11]);     // c : fp32 : 16 words
    vmovups(zm4, ptr[r11+off]); // c : fp32 : 16 words

    add(r10, sf.p[3]); // j<<5+k
    shl(r10, 1);       // 16bit
    add(r10, sf.p[0]); // &a[j*32+k];
    
    mov(eax, ptr[r10]);      // a : bf16 : 2 words
    vpbroadcastd(zm0, eax);
    mov(eax, ptr[r10+4]);    // a : bf16 : 2 words
    vpbroadcastd(zm5, eax);
    
    vdpbf16ps(zm3, zm1, zm0); // c += a*b
    vdpbf16ps(zm4, zm2, zm0); // c += a*b
    vdpbf16ps(zm3, zm6, zm5); // c += a*b
    vdpbf16ps(zm4, zm7, zm5); // c += a*b

    vmovups(ptr[r11   ],  zm3); // c : fp32 : 16 wrods
    vmovups(ptr[r11+off], zm4); // c : fp32 : 16 wrods

    inc(r9);
    cmp(r9,32);
    jb(".loop");
  }
};


struct AVXBF16_VER5 : Xbyak::CodeGenerator {
  const unsigned int of0 = 0;
  const unsigned int of1 = 64;
  const unsigned int of2 = 128;
  const unsigned int of3 = 192;
  const unsigned int of4 = of0+256;
  const unsigned int of5 = of1+256;
  const unsigned int of6 = of2+256;
  const unsigned int of7 = of3+256;

  AVXBF16_VER5() {
    align(16);
    Xbyak::util::StackFrame sf(this, 3);
    vmovups(zm0, ptr[sf.p[2]+of0]);   // c : fp32 : 16 words
    vmovups(zm1, ptr[sf.p[2]+of1]);   
    vmovups(zm2, ptr[sf.p[2]+of2]);   
    vmovups(zm3, ptr[sf.p[2]+of3]);   
    vmovups(zm4, ptr[sf.p[2]+of4]);  
    vmovups(zm5, ptr[sf.p[2]+of5]);   
    vmovups(zm6, ptr[sf.p[2]+of6]);   
    vmovups(zm7, ptr[sf.p[2]+of7]);   

    mov(eax, ptr[sf.p[0]]  ); // a : bf16 : 2 words
    vpbroadcastd(zm20, eax);
    mov(eax, ptr[sf.p[0]+4]); // a : bf16 : 2 words
    vpbroadcastd(zm21, eax);

    mov(eax, ptr[sf.p[0]+of1]  );
    vpbroadcastd(zm22, eax);
    mov(eax, ptr[sf.p[0]+of1+4]);
    vpbroadcastd(zm23, eax);

    mov(eax, ptr[sf.p[0]+of2]  ); 
    vpbroadcastd(zm24, eax);
    mov(eax, ptr[sf.p[0]+of2+4]); 
    vpbroadcastd(zm25, eax);

    mov(eax, ptr[sf.p[0]+of3]  );
    vpbroadcastd(zm26, eax);
    mov(eax, ptr[sf.p[0]+of3+4]);
    vpbroadcastd(zm27, eax);

    vmovdqu16(zm10, ptr[sf.p[1]+of0]); // b : bf16 : 32 words
    vmovdqu16(zm11, ptr[sf.p[1]+of1]); 
    vmovdqu16(zm12, ptr[sf.p[1]+of2]); 
    vmovdqu16(zm13, ptr[sf.p[1]+of3]); 
    
    vdpbf16ps(zm0, zm10, zm20); // c += a*b
    vdpbf16ps(zm0, zm12, zm21); 
    vdpbf16ps(zm1, zm11, zm20); 
    vdpbf16ps(zm1, zm13, zm21); 

    vdpbf16ps(zm2, zm10, zm22); // c += a*b
    vdpbf16ps(zm2, zm12, zm23); 
    vdpbf16ps(zm3, zm11, zm22); 
    vdpbf16ps(zm3, zm13, zm23); 

    vdpbf16ps(zm4, zm10, zm24); // c += a*b
    vdpbf16ps(zm4, zm12, zm25); 
    vdpbf16ps(zm5, zm11, zm24); 
    vdpbf16ps(zm5, zm13, zm25); 

    vdpbf16ps(zm6, zm10, zm26); // c += a*b
    vdpbf16ps(zm6, zm12, zm27); 
    vdpbf16ps(zm7, zm11, zm26); 
    vdpbf16ps(zm7, zm13, zm27); 
    
    vmovups(ptr[sf.p[2]+of0], zm0); // c : fp32 : 16 wrods
    vmovups(ptr[sf.p[2]+of1], zm1); 
    vmovups(ptr[sf.p[2]+of2], zm2); 
    vmovups(ptr[sf.p[2]+of3], zm3); 
    vmovups(ptr[sf.p[2]+of4], zm4); 
    vmovups(ptr[sf.p[2]+of5], zm5); 
    vmovups(ptr[sf.p[2]+of6], zm6); 
    vmovups(ptr[sf.p[2]+of7], zm7); 
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
    
    for(int p = 0; p < 32; p++) {
      bfloat16 tmp[4];
      for(int kk = 0; kk < 4; kk++) {
	tmp[kk] = b[(k + kk)*lda + p]; 
      }
      btmpx[2*p     ] = tmp[0];
      btmpx[2*p+1   ] = tmp[1];
      btmpx[2*p  +64] = tmp[2];
      btmpx[2*p+1+64] = tmp[3];
    }

    for(int j = 0; j < 32; j++) {
      f(&a[j*lda+k], btmpx, &c[j*lda+0]);
    }
  }
}

void mm_outer_avx_ver4x(bfloat16 *a, bfloat16 *b, float *c, const int n, const int lda)
{
  AVXBF16_VER4x code;
  auto f = code.getCode<void (*)(const void *, const void *, void *, int)>();    
  
  for(int k = 0; k < n; k += 4) {
    bfloat16 btmpx[32*2*2];
    
    for(int p = 0; p < 32; p++) {
      bfloat16 tmp[4];
      for(int kk = 0; kk < 4; kk++) {
	tmp[kk] = b[(k + kk)*lda + p]; 
      }
      btmpx[2*p     ] = tmp[0];
      btmpx[2*p+1   ] = tmp[1];
      btmpx[2*p  +64] = tmp[2];
      btmpx[2*p+1+64] = tmp[3];
    }

    f(a, btmpx, c, k);
  }
}

void mm_outer_avx_ver5(bfloat16 *a, bfloat16 *b, float *c, const int n, const int lda)
{
  AVXBF16_VER5 code;
  auto f = code.getCode<void (*)(const void *, const void *, void *)>();    
  
  for(int k = 0; k < n; k += 4) {
    bfloat16 btmpx[32*2*2];
    
    for(int p = 0; p < 32; p++) {
      bfloat16 tmp[4];
      for(int kk = 0; kk < 4; kk++) {
	tmp[kk] = b[(k + kk)*lda + p]; 
      }
      btmpx[2*p     ] = tmp[0];
      btmpx[2*p+1   ] = tmp[1];
      btmpx[2*p  +64] = tmp[2];
      btmpx[2*p+1+64] = tmp[3];
    }

    for(int j = 0; j < n; j += 4) {
      f(&a[(j  )*lda+k], btmpx, &c[(j  )*lda+0]);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

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

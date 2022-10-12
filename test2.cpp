#include <random>
#include <cstdio>
#include <cmath>
#include <cstring>

#define XBYAK_NO_OP_NAMES
#include <xbyak/xbyak.h>
#include <xbyak/xbyak_util.h>

typedef uint16_t bfloat16;

struct Loop : Xbyak::CodeGenerator {
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

  Loop() {
    align(16);
    
    Xbyak::util::StackFrame sf(this, 3);
    vmovups(zm0, ptr[sf.p[0]]);
    vmovups(zm1, ptr[sf.p[1]]);
    vmovups(zm2, ptr[sf.p[2]]);

    vdpbf16ps(zm0, zm1, zm2);
    vdpbf16ps(zm0, zm1, zm2);
    vdpbf16ps(zm0, zm1, zm2);
    vdpbf16ps(zm0, zm1, zm2);

    vmovups(ptr[sf.p[0]], zm0);
  }

  MM() {
    align(16);
    
    Xbyak::util::StackFrame sf(this, 3);
    vmovups(zm0, ptr[sf.p[0]]);

    vmovups(zm1, ptr[sf.p[1]]);

    for(int i = 0; i < 4; i++) {
      vmovups(zm2, ptr[sf.p[2] + i*64]);
      vdpbf16ps(zm0, zm1, zm2);
    }

    vmovups(ptr[sf.p[0]], zm0);
  }


};

int main()
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<> urdist(0.0, 1.0);

  try {
    Loop c;
    printf("Xbyak version=%s\n", c.getVersionString());

    float x[64];
    for(int i = 0; i < 64; i++) {
      x[i] = urdist(rd);
    }
    
    bfloat16 a[32], b[32];
    for(int i = 0; i < 32; i++) {
      a[i] = c.f2b(x[i]);
      b[i] = c.f2b(x[i+32]);
    }

    float res[16], res0[16];
    for(int i = 0; i < 16; i++) {
      res[i]  = 0.0;
    }
    
    auto f = c.getCode<void (*)(void *, const void *, const void *)>();    

    f(res, a, b);

    for(int i = 0; i < 16; i++) {
      res0[i] = 0.0;
      for(int j = 0; j < 4; j++)
	res0[i] += x[2*i]*x[2*i+32] + x[2*i+1]*x[2*i+32+1]; 
    }
    
    for(int i = 0; i < 16; i++) {
      printf("%e %e %e\n", res[i], res0[i], fabs((res[i]-res0[i])/res0[i]));
    }
    
    FILE *fp = fopen("dump.obj", "w");
    fwrite((const void *)f, c.getSize(), 1, fp);
    fclose(fp);
  } catch (std::exception& e) {
    printf("ERR:%s\n", e.what());
  } catch (...) {
    printf("unknown error\n");
  }
}


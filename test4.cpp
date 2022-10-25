#include <random>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "m1cycles.h"
#include "linux-perf-events.h"

#include "AVXBF16.hpp"

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
    float *ref, *res1, *res2, *res3, *res4, *res5, *res6;
      
    a = new bfloat16[32*32];
    b = new bfloat16[32*32];

    ref  = new float[32*32];
    res1 = new float[32*32];
    res2 = new float[32*32];
    res3 = new float[32*32];
    res4 = new float[32*32];
    res5 = new float[32*32];
    res6 = new float[32*32];    

    for(int j = 0; j < 32; j++) {
      for(int i = 0; i < 32; i++) {
	a[j*32+i] = f2b(urdist(mt));
	b[j*32+i] = f2b(urdist(mt));
	
	//	a[j*32+i] = f2b(i);
	//	if (i == j ) b[j*32+i] = f2b(1.0);
	//	else         b[j*32+i] = f2b(0.0);
      }
    }
    {
      AVXBF16_VER4x c;
      printf("Xbyak version=%s\n", c.getVersionString());

      auto f = c.getCode<void (*)(const void *, const void *, void *, int)>();    
      FILE *fp = fopen("dump.obj", "w");
      fwrite((const void *)f, c.getSize(), 1, fp);
      fclose(fp);
    }
    {
      AVXBF16_VER4 c;
      printf("Xbyak version=%s\n", c.getVersionString());

      auto f = c.getCode<void (*)(const void *, const void *, void *, int)>();    
      FILE *fp = fopen("dump2.obj", "w");
      fwrite((const void *)f, c.getSize(), 1, fp);
      fclose(fp);
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

    now = measure(LOOP, mm_outer_avx_ver4x, a, b, res5, 32, 32);
    std::cout << "outer avx ver4 "; output(now, ref, res5, ops);
    
    now = measure(LOOP, mm_outer_avx_ver5, a, b, res6, 32, 32);
    std::cout << "outer avx ver5 "; output(now, ref, res6, ops);

  } catch (std::exception& e) {
    printf("ERR:%s\n", e.what());
  } catch (...) {
    printf("unknown error\n");
  }
}


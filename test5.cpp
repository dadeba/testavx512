#include <random>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "m1cycles.h"
#include "linux-perf-events.h"

#include "AVXBF16.hpp"

void mm_blocked(bfloat16 *a, bfloat16 *b, float *c, const int n, const int lda,
		void (*func)(bfloat16 *, bfloat16 *, float *, const int, const int))
{
  const int nb = 32;

  // blocked
  for(int j = 0; j < n; j += nb) {
    for(int i = 0; i < n; i += nb) {
      bfloat16 aa[nb][nb];
      bfloat16 bb[nb][nb];
      float    cc[nb][nb];

      for(int p = 0; p < nb; p++) {
	for(int q = 0; q < nb; q++) {
	  cc[p][q] = 0.0;
	}
      }
      
      for(int k = 0; k < n; k += nb) {
	float tmp[nb*nb];

	for(int p = 0; p < nb; p++) {
	  for(int q = 0; q < nb; q++) {
	    aa[p][q] = a[(j + p)*lda + k + q];
	    bb[p][q] = b[(k + p)*lda + i + q];
	    tmp[p*nb+q] = 0.0;
	  }
	}

	func((bfloat16*)aa, (bfloat16*)bb, tmp, nb, nb);
	
	for(int p = 0; p < nb; p++) {
	  for(int q = 0; q < nb; q++) {
	    cc[p][q] += tmp[p*nb+q];
	  }
	}
      }

      for(int p = 0; p < nb; p++) {
	for(int q = 0; q < nb; q++) {
	  c[(j+p)*lda + i+q] = cc[p][q];
	}
      }
    }
  }
}

performance_counters measure(const int LOOP,
			     void (*func)(bfloat16 *, bfloat16 *, float *, const int, const int,
					  void (*func)(bfloat16 *, bfloat16 *, float *, const int, const int)),
			     void (*func_mm)(bfloat16 *, bfloat16 *, float *, const int, const int),
			     bfloat16 *a, bfloat16 *b, float *c, const int n, const int lda)
{
  performance_counters start(0.0), end(0.0), diff(0.0);
  for(int l = 0; l < LOOP; l++) {

    for(int i = 0; i < n*n; i++) c[i] = 0.0;

    start = get_counters();
    func(a, b, c, n, lda, func_mm);
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

void output(performance_counters &now, float *ref, float *res, const double ops, const int n)
{
  std::cout << now.cycles << "\t" << now.instructions << "\t"
	    << round(ops/now.cycles * 100.0)/100.0 << "\t"
    	    << ops/(now.e_time/1.0e9)/1.0e9 << "\t"
	    << L1_norm(ref, res, n) << "\n";
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

    const int n = 32*8;
    const int lda = n;
    
    a = new bfloat16[n*n];
    b = new bfloat16[n*n];

    ref  = new float[n*n];
    res1 = new float[n*n];
    res2 = new float[n*n];
    res3 = new float[n*n];
    res4 = new float[n*n];
    res5 = new float[n*n];
    res6 = new float[n*n];    

    for(int j = 0; j < n; j++) {
      for(int i = 0; i < n; i++) {
	a[j*lda+i] = f2b(urdist(mt));
	b[j*lda+i] = f2b(urdist(mt));
      }
    }

    const int LOOP = 10;
    performance_counters now;
    double ops = 2.0*pow(n, 3.0);
    
    mm_ref    (a, b, ref,  n, lda);
    //    std::cout << "ref            "; output(now, ref, ref, ops);

    now = measure(LOOP, mm_blocked, mm_outer, a, b, res1, n, lda);
    std::cout << "outer           "; output(now, ref, res1, ops, n);

    now = measure(LOOP, mm_blocked, mm_outer_avx_ver1, a, b, res2, n, lda);
    std::cout << "outer avx ver1  "; output(now, ref, res2, ops, n);
    
    now = measure(LOOP, mm_blocked, mm_outer_avx_ver2, a, b, res3, n, lda);
    std::cout << "outer avx ver2  "; output(now, ref, res3, ops, n);

    now = measure(LOOP, mm_blocked, mm_outer_avx_ver3, a, b, res4, n, lda);
    std::cout << "outer avx ver3  "; output(now, ref, res4, ops, n);

    now = measure(LOOP, mm_blocked, mm_outer_avx_ver4, a, b, res5, n, lda);
    std::cout << "outer avx ver4  "; output(now, ref, res5, ops, n);

    now = measure(LOOP, mm_blocked, mm_outer_avx_ver4x, a, b, res5, n, lda);
    std::cout << "outer avx ver4x "; output(now, ref, res5, ops, n);

    now = measure(LOOP, mm_blocked, mm_outer_avx_ver5, a, b, res6, n, lda);
    std::cout << "outer avx ver5  "; output(now, ref, res6, ops, n);

  } catch (std::exception& e) {
    printf("ERR:%s\n", e.what());
  } catch (...) {
    printf("unknown error\n");
  }
}


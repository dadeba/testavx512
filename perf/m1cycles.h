// source: https://lemire.me/blog/2021/03/24/counting-cycles-and-instructions-on-the-apple-m1-processor/
//         https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/tree/master/2021/03/24

#ifndef M1CYCLES_H
#define M1CYCLES_H

#include <cstdint>
#include <chrono>

struct performance_counters {
  double cycles;
  double branches;
  double missed_branches;
  double instructions;
  std::chrono::steady_clock::time_point time_point;
  double e_time;
  
  performance_counters(uint64_t c, uint64_t b, uint64_t m, uint64_t i,
		       std::chrono::steady_clock::time_point ti)
    : cycles(c), branches(b), missed_branches(m), instructions(i), time_point(ti) {}
  
  performance_counters(uint64_t c, uint64_t b, uint64_t m, uint64_t i)
    : cycles(c), branches(b), missed_branches(m), instructions(i) {}
  performance_counters(double c, double b, double m, double i, double t)
    : cycles(c), branches(b), missed_branches(m), instructions(i), e_time(t) {}
  performance_counters(double c, double b, double m, double i,
 		       std::chrono::steady_clock::time_point ti)
    : cycles(c), branches(b), missed_branches(m), instructions(i), time_point(ti) {}

  performance_counters(double init)
      : cycles(init), branches(init), missed_branches(init),
    instructions(init) {}
  performance_counters(void)
    : cycles(0.0), branches(0.0), missed_branches(0.0),
    instructions(0.0) {}

  inline performance_counters &operator-=(const performance_counters &other) {
    cycles -= other.cycles;
    branches -= other.branches;
    missed_branches -= other.missed_branches;
    instructions -= other.instructions;
    e_time = std::chrono::duration_cast<std::chrono::nanoseconds>(time_point - other.time_point).count();
    return *this;
  }

  inline performance_counters &min(const performance_counters &other) {
    cycles = other.cycles < cycles ? other.cycles : cycles;
    branches = other.branches < branches ? other.branches : branches;
    missed_branches = other.missed_branches < missed_branches
                          ? other.missed_branches
                          : missed_branches;
    instructions =
        other.instructions < instructions ? other.instructions : instructions;
    return *this;
  }

  inline performance_counters &operator+=(const performance_counters &other) {
    cycles += other.cycles;
    branches += other.branches;
    missed_branches += other.missed_branches;
    instructions += other.instructions;
    e_time += other.e_time;
    return *this;
  }

  inline performance_counters &operator/=(double numerator) {
    cycles /= numerator;
    branches /= numerator;
    missed_branches /= numerator;
    instructions /= numerator;
    e_time /= numerator;
    return *this;
  }
};

inline performance_counters operator-(const performance_counters &a,
                                      const performance_counters &b) {

  double diff_time = std::chrono::duration_cast<std::chrono::nanoseconds>(a.time_point - b.time_point).count();
    
  return performance_counters(a.cycles - b.cycles,
			      a.branches - b.branches,
                              a.missed_branches - b.missed_branches,
                              a.instructions - b.instructions,
			      diff_time
			      );
}

void setup_performance_counters();
#endif

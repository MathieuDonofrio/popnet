#include <benchmark/benchmark.h>

static void skeleton(benchmark::State& state)
{
  for (auto _ : state)
  {
    int x;

    // TODO

    benchmark::DoNotOptimize(x);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(skeleton);

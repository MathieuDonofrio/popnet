#include "reduce.h"

#include "bench_helpers.h"

template <bool naive> static void reduce_sum(benchmark::State& state)
{
  const size_t rows = state.range(0), columns = state.range(1);
  auto a = bench_input(rows * columns);
  std::vector<float> b(rows);
  popnet_tensor src{.shape = {{1, 1, rows, columns}}, .data = a.data(), .dtype = popnet_dtype_f32};
  popnet_tensor dst{.shape = {{1, 1, rows, 1}}, .data = b.data(), .dtype = popnet_dtype_f32};
  dst.src = {&src};
  benchmark::DoNotOptimize(a.data());
  benchmark::DoNotOptimize(b.data());

  for (auto _ : state)
  {
    if constexpr (naive)
    {
      for (size_t row = 0; row < rows; ++row)
      {
        float sum = 0;
        for (size_t col = 0; col < columns; ++col)
          sum += a[row * columns + col];
        b[row] = sum;
      }
    }
    else
      popnet_compute_reduce_sum(&dst);
    benchmark::ClobberMemory();
  }
}

// Full reduction, SIMD tail, and many contiguous row reductions.
BENCHMARK_TEMPLATE(reduce_sum, false)
    ->Name("f32/reduce_sum/popnet")
    ->Args({1, 31})
    ->Args({1, 1048576})
    ->Args({1024, 257});
BENCHMARK_TEMPLATE(reduce_sum, true)
    ->Name("f32/reduce_sum/naive")
    ->Args({1, 31})
    ->Args({1, 1048576})
    ->Args({1024, 257});

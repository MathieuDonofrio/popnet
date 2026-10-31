#include "reduce.h"

#include "bench_helpers.h"
#include "ops.h"

template <bool mean> static void reduce_axes(benchmark::State& state)
{
  const popnet_shape shape{{static_cast<size_t>(state.range(0)), static_cast<size_t>(state.range(1)),
                            static_cast<size_t>(state.range(2)), static_cast<size_t>(state.range(3))}};
  const auto mask = static_cast<uint8_t>(state.range(4));
  const popnet_shape output_shape = popnet_reduce_shape(shape, mask);
  auto input = bench_input(popnet_numel(shape));
  std::vector<float> output(popnet_numel(output_shape));
  popnet_tensor src{.shape = shape, .data = input.data(), .dtype = popnet_dtype_f32};
  popnet_tensor dst{.shape = output_shape, .data = output.data(), .src = {&src}, .dtype = popnet_dtype_f32};
  benchmark::DoNotOptimize(input.data());
  benchmark::DoNotOptimize(output.data());

  for (auto _ : state)
  {
    if constexpr (mean) popnet_compute_reduce_mean(&dst);
    else
      popnet_compute_reduce_sum(&dst);
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE(reduce_axes, false)
    ->Name("f32/reduce_sum/axes/popnet")
    ->Args({1, 1, 16, 31, 4})
    ->Args({1, 1, 1024, 1024, 4})
    ->Args({2, 3, 4, 5, 6})
    ->Args({4, 16, 128, 32, 6});
BENCHMARK_TEMPLATE(reduce_axes, true)
    ->Name("f32/reduce_mean/axes/popnet")
    ->Args({2, 3, 4, 5, 6})
    ->Args({4, 16, 128, 32, 6});

template <bool naive, bool mean = false> static void reduce(benchmark::State& state)
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
        b[row] = mean ? sum / static_cast<float>(columns) : sum;
      }
    }
    else if constexpr (mean)
      popnet_compute_reduce_mean(&dst);
    else
      popnet_compute_reduce_sum(&dst);
    benchmark::ClobberMemory();
  }
}

// Full reduction, SIMD tail, and many contiguous row reductions.
BENCHMARK_TEMPLATE(reduce, false)->Name("f32/reduce_sum/popnet")->Args({1, 31})->Args({1, 1048576})->Args({1024, 257});
BENCHMARK_TEMPLATE(reduce, true)->Name("f32/reduce_sum/naive")->Args({1, 31})->Args({1, 1048576})->Args({1024, 257});

BENCHMARK_TEMPLATE(reduce, false, true)
    ->Name("f32/reduce_mean/popnet")
    ->Args({17, 31})
    ->Args({1, 1048576})
    ->Args({1024, 257});
BENCHMARK_TEMPLATE(reduce, true, true)
    ->Name("f32/reduce_mean/naive")
    ->Args({17, 31})
    ->Args({1, 1048576})
    ->Args({1024, 257});

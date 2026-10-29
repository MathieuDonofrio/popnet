#include "linalg.h"

#include "bench_helpers.h"

template <bool naive> static void matmul(benchmark::State& state)
{
  const size_t batch = state.range(0), m = state.range(1), n = state.range(2), k = state.range(3);
  auto a = bench_input(batch * m * k);
  auto b = bench_input(n * k);
  std::vector<float> c(batch * m * n);
  popnet_tensor lhs{.shape = {{batch, 1, m, k}}, .data = a.data(), .dtype = popnet_dtype_f32};
  popnet_tensor rhs{.shape = {{1, 1, n, k}}, .data = b.data(), .dtype = popnet_dtype_f32};
  popnet_tensor dst{.shape = {{batch, 1, m, n}}, .data = c.data(), .dtype = popnet_dtype_f32};
  dst.src = {&lhs, &rhs};
  benchmark::DoNotOptimize(a.data());
  benchmark::DoNotOptimize(b.data());
  benchmark::DoNotOptimize(c.data());

  for (auto _ : state)
  {
    if constexpr (naive)
    {
      // Both paths compute A * transpose(B), sharing B across batches.
      for (size_t p = 0; p < batch; ++p)
        for (size_t i = 0; i < m; ++i)
          for (size_t j = 0; j < n; ++j)
          {
            float sum = 0;
            for (size_t t = 0; t < k; ++t)
              sum += a[(p * m + i) * k + t] * b[j * k + t];
            c[(p * m + i) * n + j] = sum;
          }
    }
    else
      popnet_compute_matmul(&dst);
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE(matmul, false)
    ->Name("f32/matmul/popnet")
    ->Args({1, 8, 8, 8})
    ->Args({1, 256, 256, 256})
    ->Args({4, 31, 65, 127});
BENCHMARK_TEMPLATE(matmul, true)
    ->Name("f32/matmul/naive")
    ->Args({1, 8, 8, 8})
    ->Args({1, 256, 256, 256})
    ->Args({4, 31, 65, 127});

template <bool naive> static void transpose(benchmark::State& state)
{
  const size_t batch = state.range(0), m = state.range(1), n = state.range(2);
  auto a = bench_input(batch * m * n);
  std::vector<float> b(a.size());
  popnet_tensor src{.shape = {{batch, 1, m, n}}, .data = a.data(), .dtype = popnet_dtype_f32};
  popnet_tensor dst{.shape = {{batch, 1, n, m}}, .data = b.data(), .dtype = popnet_dtype_f32};
  dst.src = {&src};
  benchmark::DoNotOptimize(a.data());
  benchmark::DoNotOptimize(b.data());

  for (auto _ : state)
  {
    if constexpr (naive)
    {
      for (size_t p = 0; p < batch; ++p)
        for (size_t i = 0; i < m; ++i)
          for (size_t j = 0; j < n; ++j)
            b[p * m * n + j * m + i] = a[p * m * n + i * n + j];
    }
    else
      popnet_compute_transpose(&dst);
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE(transpose, false)
    ->Name("f32/transpose/popnet")
    ->Args({1, 8, 8})
    ->Args({1, 1024, 1024})
    ->Args({4, 63, 129});
BENCHMARK_TEMPLATE(transpose, true)
    ->Name("f32/transpose/naive")
    ->Args({1, 8, 8})
    ->Args({1, 1024, 1024})
    ->Args({4, 63, 129});

#include "elemwise.h"

#include <cmath>

#include "bench_helpers.h"

enum class add_path
{
  matching,
  left_scalar,
  right_scalar,
  row,
  column
};

template <add_path path, bool naive> static void add(benchmark::State& state)
{
  const size_t rows = state.range(0), cols = state.range(1);
  const popnet_shape shape{{1, 1, rows, cols}};
  const popnet_shape lhs_shape = path == add_path::left_scalar ? popnet_shape{{1, 1, 1, 1}} : shape;
  const popnet_shape rhs_shape = path == add_path::right_scalar ? popnet_shape{{1, 1, 1, 1}}
                                 : path == add_path::row        ? popnet_shape{{1, 1, 1, cols}}
                                 : path == add_path::column     ? popnet_shape{{1, 1, rows, 1}}
                                                                : shape;
  auto a = bench_input(popnet_numel(lhs_shape));
  auto b = bench_input(popnet_numel(rhs_shape));
  std::vector<float> c(rows * cols);
  popnet_tensor lhs{.shape = lhs_shape, .data = a.data(), .dtype = popnet_dtype_f32};
  popnet_tensor rhs{.shape = rhs_shape, .data = b.data(), .dtype = popnet_dtype_f32};
  popnet_tensor dst{.shape = shape, .data = c.data(), .dtype = popnet_dtype_f32};
  dst.src = {&lhs, &rhs};
  benchmark::DoNotOptimize(a.data());
  benchmark::DoNotOptimize(b.data());
  benchmark::DoNotOptimize(c.data());

  for (auto _ : state)
  {
    if constexpr (naive)
    {
      if constexpr (path == add_path::matching)
      {
        for (size_t i = 0; i < c.size(); ++i)
          c[i] = a[i] + b[i];
      }
      else if constexpr (path == add_path::left_scalar)
      {
        const float scalar = a[0];
        for (size_t i = 0; i < c.size(); ++i)
          c[i] = scalar + b[i];
      }
      else if constexpr (path == add_path::right_scalar)
      {
        const float scalar = b[0];
        for (size_t i = 0; i < c.size(); ++i)
          c[i] = a[i] + scalar;
      }
      else
      {
        for (size_t row = 0; row < rows; ++row)
        {
          const float* input = a.data() + row * cols;
          float* output = c.data() + row * cols;
          if constexpr (path == add_path::row)
          {
            for (size_t col = 0; col < cols; ++col)
              output[col] = input[col] + b[col];
          }
          else
          {
            const float scalar = b[row];
            for (size_t col = 0; col < cols; ++col)
              output[col] = input[col] + scalar;
          }
        }
      }
    }
    else
      popnet_compute_add(&dst);
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE(add, add_path::matching, false)->Name("f32/add/matching/popnet")->Args({1, 31})->Args({1024, 1024});
BENCHMARK_TEMPLATE(add, add_path::matching, true)->Name("f32/add/matching/naive")->Args({1, 31})->Args({1024, 1024});
BENCHMARK_TEMPLATE(add, add_path::left_scalar, false)
    ->Name("f32/add/left_scalar/popnet")
    ->Args({1, 31})
    ->Args({1024, 1024});
BENCHMARK_TEMPLATE(add, add_path::left_scalar, true)
    ->Name("f32/add/left_scalar/naive")
    ->Args({1, 31})
    ->Args({1024, 1024});
BENCHMARK_TEMPLATE(add, add_path::right_scalar, false)
    ->Name("f32/add/right_scalar/popnet")
    ->Args({1, 31})
    ->Args({1024, 1024});
BENCHMARK_TEMPLATE(add, add_path::right_scalar, true)
    ->Name("f32/add/right_scalar/naive")
    ->Args({1, 31})
    ->Args({1024, 1024});
BENCHMARK_TEMPLATE(add, add_path::row, false)->Name("f32/add/row/popnet")->Args({1024, 257});
BENCHMARK_TEMPLATE(add, add_path::row, true)->Name("f32/add/row/naive")->Args({1024, 257});
BENCHMARK_TEMPLATE(add, add_path::column, false)->Name("f32/add/column/popnet")->Args({1024, 257});
BENCHMARK_TEMPLATE(add, add_path::column, true)->Name("f32/add/column/naive")->Args({1024, 257});

template <bool naive> static void square_root(benchmark::State& state)
{
  const size_t n = state.range(0);
  auto a = bench_input(n);
  std::vector<float> b(n);
  popnet_tensor src{.shape = {{1, 1, 1, n}}, .data = a.data(), .dtype = popnet_dtype_f32};
  popnet_tensor dst{.shape = src.shape, .data = b.data(), .dtype = popnet_dtype_f32};
  dst.src = {&src};
  benchmark::DoNotOptimize(a.data());
  benchmark::DoNotOptimize(b.data());

  for (auto _ : state)
  {
    if constexpr (naive)
    {
      for (size_t i = 0; i < n; ++i)
        b[i] = std::sqrt(a[i]);
    }
    else
      popnet_compute_sqrt(&dst);
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE(square_root, false)->Name("f32/square_root/popnet")->Arg(31)->Arg(1048576);
BENCHMARK_TEMPLATE(square_root, true)->Name("f32/square_root/naive")->Arg(31)->Arg(1048576);

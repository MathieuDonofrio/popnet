#pragma once

#include "dtype_traits.h"

// Pointer tag for a scalar reused across lanes and iterations; no broadcast object is stored.
struct broadcast
{};
struct row_broadcast
{};
struct column_broadcast
{};

template <popnet_dtype dtype, typename T> POPNET_INLINE auto maybe_broadcast_vload(const T* p)
{
  using dtype_traits = dtype_traits<dtype>;

  if constexpr (std::same_as<T, broadcast>)
  {
    return dtype_traits::broadcast(dtype_traits::load(p));
  }
  else
  {
    return dtype_traits::vload(p);
  }
}

template <popnet_dtype dtype, typename T>
POPNET_INLINE auto maybe_broadcast_vload(const T* p, typename dtype_traits<dtype>::mask_t m)
{
  using dtype_traits = dtype_traits<dtype>;

  if constexpr (std::same_as<T, broadcast>)
  {
    return dtype_traits::broadcast(dtype_traits::load(p));
  }
  else
  {
    return dtype_traits::vload(p, m);
  }
}

template <popnet_dtype dtype, typename T> POPNET_INLINE const T* maybe_broadcast_advance(const T* p, size_t n)
{
  using dtype_traits = dtype_traits<dtype>;

  if constexpr (std::same_as<T, broadcast>)
  {
    return p;
  }
  else
  {
    return static_cast<const T*>(dtype_traits::advance(p, n));
  }
}

template <popnet_dtype dtype, typename T>
POPNET_INLINE auto maybe_broadcast_advance_2d(const T* p, size_t row, size_t col, size_t cols) noexcept
{
  if constexpr (std::same_as<T, row_broadcast>)
  {
    return static_cast<const broadcast*>(maybe_broadcast_advance<dtype>(p, row));
  }
  else if constexpr (std::same_as<T, column_broadcast>)
  {
    return maybe_broadcast_advance<dtype>(p, col);
  }
  else
  {
    return maybe_broadcast_advance<dtype>(p, row * cols + col);
  }
}

template <class T, class... Ts>
POPNET_INLINE constexpr T&& pack_first_element([[maybe_unused]] T&& t, [[maybe_unused]] Ts&&... ts) noexcept
{
  if constexpr (std::same_as<T, broadcast> && sizeof...(ts) > 0)
  {
    return pack_first_element(std::forward<Ts>(ts)...);
  }
  else
  {
    return std::forward<T>(t);
  }
}

template <bool inplace, auto op, popnet_dtype dtype, typename... T>
POPNET_INLINE static void simd_map_impl(void* output, size_t n, const T*... inputs) noexcept
{
  using dtype_traits = dtype_traits<dtype>;

  if constexpr (dtype_traits::cpu_execution_model == cpu_execution_model::fixed)
  {
    constexpr size_t vector_length = dtype_traits::vector_length();

    const size_t tail = n % vector_length;
    const size_t body = n - tail;

    const void* const output_body_end = dtype_traits::advance(output, body);

    while (output != output_body_end)
    {
      if constexpr (inplace)
        dtype_traits::vstore(output, op(dtype_traits::vload(output), maybe_broadcast_vload<dtype>(inputs)...));
      else
        dtype_traits::vstore(output, op(maybe_broadcast_vload<dtype>(inputs)...));

      output = dtype_traits::advance(output, vector_length);
      ((inputs = maybe_broadcast_advance<dtype>(inputs, vector_length)), ...);
    }

    if (tail)
    {
      const typename dtype_traits::mask_t m = (1 << tail) - 1;

      if constexpr (inplace)
        dtype_traits::vstore(output, m, op(dtype_traits::vload(output, m), maybe_broadcast_vload<dtype>(inputs, m)...));
      else
        dtype_traits::vstore(output, m, op(maybe_broadcast_vload<dtype>(inputs, m)...));
    }
  }
  else if constexpr (dtype_traits::cpu_execution_model == cpu_execution_model::scalar)
  {
    const void* const output_end = dtype_traits::advance(output, n);

    while (output != output_end)
    {
      if constexpr (inplace)
        dtype_traits::store(output, op(dtype_traits::load(output), dtype_traits::load(inputs)...));
      else
        dtype_traits::store(output, op(dtype_traits::load(inputs)...));

      output = dtype_traits::advance(output, 1);
      ((inputs = maybe_broadcast_advance<dtype>(inputs, 1)), ...);
    }
  }
  else
  {
    POPNET_UNSUPPORTED_STATIC("SIMD algorithms require a fixed or scalar CPU execution model");
  }
}

template <auto op, popnet_dtype dtype, typename... T>
POPNET_INLINE static void simd_map(void* POPNET_RESTRICT output, size_t n, const T* POPNET_RESTRICT... inputs) noexcept
{
  simd_map_impl<false, op, dtype>(output, n, inputs...);
}

// The output supplies the first operand. Additional inputs must not overlap the output.
template <auto op, popnet_dtype dtype, typename... T>
POPNET_INLINE static void simd_map_inplace(void* POPNET_RESTRICT output, size_t n,
                                         const T* POPNET_RESTRICT... inputs) noexcept
{
  simd_map_impl<true, op, dtype>(output, n, inputs...);
}

template <auto state_init_op, auto merge_op, auto horizontal_op, popnet_dtype dtype, typename... T>
requires(sizeof...(T) >= 1)
POPNET_INLINE static auto simd_reduce(size_t n, const T* POPNET_RESTRICT... inputs) noexcept
{
  using dtype_traits = dtype_traits<dtype>;

  auto s = state_init_op();

  if constexpr (dtype_traits::cpu_execution_model == cpu_execution_model::fixed)
  {
    constexpr size_t vector_length = dtype_traits::vector_length();

    const size_t tail = n % vector_length;
    const size_t body = n - tail;

    const void* const first_input_end = dtype_traits::advance(pack_first_element(inputs...), body);

    while (pack_first_element(inputs...) != first_input_end)
    {
      s = merge_op(maybe_broadcast_vload<dtype>(inputs)..., s);

      ((inputs = maybe_broadcast_advance<dtype>(inputs, vector_length)), ...);
    }

    if (tail)
    {
      const typename dtype_traits::mask_t m = (1 << tail) - 1;

      // Preserve inactive accumulator lanes: zero-filled masked loads are not neutral for every reduction.
      s = dtype_traits::blend(s, merge_op(maybe_broadcast_vload<dtype>(inputs, m)..., s), m);
    }
  }
  else if constexpr (dtype_traits::cpu_execution_model == cpu_execution_model::scalar)
  {
    const void* const first_input_end = dtype_traits::advance(pack_first_element(inputs...), n);

    while (pack_first_element(inputs...) != first_input_end)
    {
      s = merge_op(dtype_traits::load(inputs)..., s);

      ((inputs = maybe_broadcast_advance<dtype>(inputs, 1)), ...);
    }
  }
  else
  {
    POPNET_UNSUPPORTED_STATIC("SIMD algorithms require a fixed or scalar CPU execution model");
  }

  if constexpr (std::is_invocable_v<decltype(horizontal_op), decltype(s), size_t>)
  {
    return horizontal_op(s, n);
  }
  else
  {
    return horizontal_op(s);
  }
}

template <auto state_init_op, auto merge_op, popnet_dtype dtype, typename... T>
requires(sizeof...(T) >= 1)
POPNET_INLINE static void simd_reduce_rows(void* POPNET_RESTRICT output, size_t rows, size_t cols,
                                           const T* POPNET_RESTRICT... inputs) noexcept
{
  using traits = dtype_traits<dtype>;

  if constexpr (traits::cpu_execution_model == cpu_execution_model::fixed)
  {
    constexpr size_t vector_length = traits::vector_length();

    const size_t tail = cols % vector_length;
    const size_t body = cols - tail;

    for (size_t col = 0; col < body; col += vector_length)
    {
      auto s = state_init_op();

      for (size_t row = 0; row < rows; ++row)
      {
        s = merge_op(maybe_broadcast_vload<dtype>(maybe_broadcast_advance_2d<dtype>(inputs, row, col, cols))..., s);
      }

      traits::vstore(traits::advance(output, col), s);
    }

    if (tail)
    {
      const typename traits::mask_t m = (uint64_t{1} << tail) - 1;

      auto s = state_init_op();

      for (size_t row = 0; row < rows; ++row)
      {
        s = traits::blend(
            s,
            merge_op(maybe_broadcast_vload<dtype>(maybe_broadcast_advance_2d<dtype>(inputs, row, body, cols), m)..., s),
            m);
      }

      traits::vstore(traits::advance(output, body), m, s);
    }
  }
  else if constexpr (traits::cpu_execution_model == cpu_execution_model::scalar)
  {
    for (size_t col = 0; col < cols; ++col)
    {
      auto s = state_init_op();

      for (size_t row = 0; row < rows; ++row)
      {
        s = merge_op(traits::load(maybe_broadcast_advance_2d<dtype>(inputs, row, col, cols))..., s);
      }

      traits::store(traits::advance(output, col), s);
    }
  }
  else
  {
    POPNET_UNSUPPORTED_STATIC("SIMD algorithms require a fixed or scalar CPU execution model");
  }
}

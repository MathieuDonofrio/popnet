#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "simd_traits.h"

// Pointer tag for a scalar reused across lanes and iterations; no broadcast object is stored.
struct broadcast
{};

inline constexpr auto identity_finalize = [](auto value, size_t) { return value; };

template <popnet_dtype dtype, typename T> POPNET_INLINE auto maybe_broadcast_vload(const T* p)
{
  using traits = simd_traits<dtype>;

  if constexpr (std::same_as<T, broadcast>)
  {
    return traits::broadcast(traits::load(p));
  }
  else
  {
    return traits::vload(p);
  }
}

template <popnet_dtype dtype, typename T>
POPNET_INLINE auto maybe_broadcast_vload(const T* p, typename simd_traits<dtype>::mask_t m)
{
  using traits = simd_traits<dtype>;

  if constexpr (std::same_as<T, broadcast>)
  {
    return traits::broadcast(traits::load(p));
  }
  else
  {
    return traits::vload(p, m);
  }
}

template <popnet_dtype dtype, typename T> POPNET_INLINE const T* maybe_broadcast_advance(const T* p, size_t n)
{
  using traits = simd_traits<dtype>;

  if constexpr (std::same_as<T, broadcast>)
  {
    return p;
  }
  else
  {
    return static_cast<const T*>(traits::advance(p, n));
  }
}

template <auto op, popnet_dtype dtype, typename... T>
POPNET_INLINE static void simd_map(void* POPNET_RESTRICT output, size_t n, const T* POPNET_RESTRICT... inputs) noexcept
{
  using traits = simd_traits<dtype>;

  constexpr bool variable = traits::cpu_execution_model == cpu_execution_model::variable;
  const size_t vector_length = traits::vector_length();

  // Count down full vectors so scalable widths need no remainder calculation.
  size_t remaining = variable ? n : n % vector_length;
  const void* const output_body_end = traits::advance(output, n - remaining);

  while (variable ? remaining >= vector_length : output != output_body_end)
  {
    traits::vstore(output, op(maybe_broadcast_vload<dtype>(inputs)...));

    output = traits::advance(output, vector_length);
    ((inputs = maybe_broadcast_advance<dtype>(inputs, vector_length)), ...);
    if constexpr (variable) remaining -= vector_length;
  }

  if constexpr (traits::cpu_execution_model != cpu_execution_model::scalar)
  {
    if (remaining)
    {
      const auto m = traits::tail_mask(remaining);
      traits::vstore(output, m, op(maybe_broadcast_vload<dtype>(inputs, m)...));
    }
  }
}

template <auto state_init_op, auto merge_op, auto horizontal_op, popnet_dtype dtype, auto finalize_op, typename... T>
requires(sizeof...(T) >= 1)
POPNET_INLINE static auto simd_reduce(size_t n, const T* POPNET_RESTRICT... inputs) noexcept
{
  using traits = simd_traits<dtype>;

  constexpr bool variable = traits::cpu_execution_model == cpu_execution_model::variable;
  const size_t vector_length = traits::vector_length();
  size_t remaining = variable ? n : n % vector_length;
  const size_t body = n - remaining;
  auto s = state_init_op();

  for (size_t i = 0; variable ? remaining >= vector_length : i < body; i += vector_length)
  {
    s = merge_op(maybe_broadcast_vload<dtype>(inputs)..., s);

    ((inputs = maybe_broadcast_advance<dtype>(inputs, vector_length)), ...);
    if constexpr (variable) remaining -= vector_length;
  }

  if constexpr (traits::cpu_execution_model != cpu_execution_model::scalar)
  {
    if (remaining)
    {
      const auto m = traits::tail_mask(remaining);

      // Preserve inactive accumulator lanes: zero-filled masked loads are not neutral for every reduction.
      s = traits::blend(s, merge_op(maybe_broadcast_vload<dtype>(inputs, m)..., s), m);
    }
  }

  return finalize_op(horizontal_op(s), n);
}

template <auto state_init_op, auto merge_op, popnet_dtype dtype, auto finalize_op>
POPNET_INLINE static void simd_reduce_rows(void* POPNET_RESTRICT output, size_t rows, size_t cols,
                                          const typename simd_traits<dtype>::scalar_t* POPNET_RESTRICT input) noexcept
{
  using traits = simd_traits<dtype>;

  constexpr bool variable = traits::cpu_execution_model == cpu_execution_model::variable;
  const size_t vector_length = traits::vector_length();
  const size_t body = variable ? cols : cols - cols % vector_length;
  size_t col = 0;

  for (; variable ? cols - col >= vector_length : col < body; col += vector_length)
  {
    auto s = state_init_op();

    for (size_t row = 0; row < rows; ++row)
    {
      s = merge_op(traits::vload(input + row * cols + col), s);
    }

    traits::vstore(traits::advance(output, col), finalize_op(s, rows));
  }

  if constexpr (traits::cpu_execution_model != cpu_execution_model::scalar)
  {
    if (col < cols)
    {
      const auto m = traits::tail_mask(cols - col);
      auto s = state_init_op();

      // Each lane is an independent output, and the masked store skips inactive lanes.
      for (size_t row = 0; row < rows; ++row)
      {
        s = merge_op(traits::vload(input + row * cols + col, m), s);
      }

      traits::vstore(traits::advance(output, col), m, finalize_op(s, rows));
    }
  }
}

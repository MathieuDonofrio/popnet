#pragma once

#include "dtype_traits.h"

struct broadcast
{};

template <popnet_dtype dtype, typename T> auto maybe_broadcast_vload(const T* p)
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

template <popnet_dtype dtype, typename T> auto maybe_broadcast_vload(const T* p, typename dtype_traits<dtype>::mask_t m)
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

template <popnet_dtype dtype, typename T> const T* maybe_broadcast_advance(const T* p, size_t n)
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

template <class T, class... Ts>
constexpr T&& pack_first_element([[maybe_unused]] T&& t, [[maybe_unused]] Ts&&... ts) noexcept
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

template <auto op, popnet_dtype dtype, typename... T>
POPNET_INLINE static void simd_map(void* POPNET_RESTRICT output, size_t n, const T* POPNET_RESTRICT... inputs) noexcept
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
      const auto value = op(maybe_broadcast_vload<dtype>(inputs)...);

      dtype_traits::vstore(output, value);

      output = dtype_traits::advance(output, vector_length);
      ((inputs = maybe_broadcast_advance<dtype>(inputs, vector_length)), ...);
    }

    if (tail)
    {
      const dtype_traits::mask_t m = (1 << tail) - 1;

      const auto v = op(maybe_broadcast_vload<dtype>(inputs, m)...);

      dtype_traits::vstore(output, m, v);
    }
  }
  else if constexpr (dtype_traits::cpu_execution_model == cpu_execution_model::scalar)
  {
    const void* const output_end = dtype_traits::advance(output, n);

    while (output != output_end)
    {
      const auto value = op(dtype_traits::load(inputs)...);

      dtype_traits::store(output, value);

      output = dtype_traits::advance(output, 1);
      ((inputs = maybe_broadcast_advance<dtype>(inputs, 1)), ...);
    }
  }
  else
  {
    static_assert([]() { return false; }, "SIMD cpu_execution_model not supported yet");
  }
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
      const dtype_traits::mask_t m = (1 << tail) - 1;

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
    static_assert([]() { return false; }, "SIMD cpu_execution_model not supported yet");
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
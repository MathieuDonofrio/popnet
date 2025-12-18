#include "linalg.h"

#include <doctest/doctest.h>
#include <numeric>
#include <vector>

#include "dtype_traits.h"
#include "ops.h"
#include "tensor.h"
#include "test_utils.h"

template <popnet_dtype dtype> static auto transpose_ref(const auto& src, size_t outer_size, size_t n, size_t m)
{
  using dtype_traits = dtype_traits<dtype>;
  using scalar_t = typename dtype_traits::scalar_t;

  const size_t plane_size = n * m;

  std::vector<scalar_t> dst;
  dst.resize(src.size());

  for (size_t outer = 0; outer < outer_size; ++outer)
  {
    const auto src_offset = outer * plane_size;
    const auto dst_offset = outer * plane_size;

    for (size_t i = 0; i < n; ++i)
    {
      for (size_t j = 0; j < m; ++j)
      {
        dst[dst_offset + j * n + i] = src[src_offset + i * m + j];
      }
    }
  }

  return dst;
}

template <popnet_dtype dtype>
static auto matmul_ref(const auto& a, const auto& b, size_t outer_size, size_t m, size_t n, size_t k)
{
  using dtype_traits = dtype_traits<dtype>;
  using scalar_t = typename dtype_traits::scalar_t;

  const size_t a_plane_size = m * k;
  const size_t c_plane_size = m * n;

  std::vector<scalar_t> c;
  c.resize(outer_size * c_plane_size);

  for (size_t outer = 0; outer < outer_size; ++outer)
  {
    const size_t a_offset = outer * a_plane_size;
    const size_t c_offset = outer * c_plane_size;

    for (size_t i = 0; i < m; ++i)
    {
      for (size_t j = 0; j < n; ++j)
      {
        scalar_t sum{};

        for (size_t kk = 0; kk < k; ++kk)
        {
          sum += a[a_offset + i * k + kk] * b[j * k + kk];
        }

        c[c_offset + i * n + j] = sum;
      }
    }
  }

  return c;
}

TEST_CASE("matmul")
{
  SUBCASE("[[1,2,3; 4,5,6]] x [[7,8,9; 10,11,12]]^T - f32")
  {
    popnet_shape a_shape = {{1, 1, 2, 3}}; // (outer=1, m=2, k=3)
    popnet_shape b_shape = {{1, 1, 2, 3}}; // (n=2, k=3)
    popnet_shape c_shape = {{1, 1, 2, 2}}; // (m=2, n=2)

    std::vector<float> a_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> b_data = {7, 8, 9, 10, 11, 12};

    std::vector<float> expected_data = {1 * 7 + 2 * 8 + 3 * 9, 1 * 10 + 2 * 11 + 3 * 12, 4 * 7 + 5 * 8 + 6 * 9,
                                        4 * 10 + 5 * 11 + 6 * 12};

    auto ref_data = matmul_ref<popnet_dtype_f32>(a_data, b_data, 1, 2, 2, 3);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* a = popnet_empty(popnet_dtype_f32, a_shape);
    popnet_tensor* b = popnet_empty(popnet_dtype_f32, b_shape);
    popnet_tensor* c = popnet_empty(popnet_dtype_f32, c_shape);

    REQUIRE(popnet_set_data(a, a_data.data(), popnet_dtype_f32, 0, popnet_numel(a_shape)));
    REQUIRE(popnet_set_data(b, b_data.data(), popnet_dtype_f32, 0, popnet_numel(b_shape)));

    c->src[0] = a;
    c->src[1] = b;
    popnet_compute_matmul(c);

    std::vector<float> output_data(popnet_numel(c_shape));
    REQUIRE(popnet_get_data(c, output_data.data(), popnet_dtype_f32, 0, popnet_numel(c_shape)));
    for (size_t i = 0; i < output_data.size(); ++i)
    {
      CHECK(output_data[i] == doctest::Approx(expected_data[i]).epsilon(1e-5f).scale(1.0f));
    }
  }

  SUBCASE("[[1,2; 3,4; 5,6]] x [[1,1; 1,1; 1,1]]^T - f32")
  {
    popnet_shape a_shape = {{1, 1, 3, 2}}; // (m=3, k=2)
    popnet_shape b_shape = {{1, 1, 3, 2}}; // (n=3, k=2)
    popnet_shape c_shape = {{1, 1, 3, 3}}; // (m=3, n=3)

    std::vector<float> a_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> b_data = {1, 1, 1, 1, 1, 1};

    std::vector<float> expected_data = {3, 3, 3, 7, 7, 7, 11, 11, 11};

    auto ref_data = matmul_ref<popnet_dtype_f32>(a_data, b_data, 1, 3, 3, 2);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* a = popnet_empty(popnet_dtype_f32, a_shape);
    popnet_tensor* b = popnet_empty(popnet_dtype_f32, b_shape);
    popnet_tensor* c = popnet_empty(popnet_dtype_f32, c_shape);

    REQUIRE(popnet_set_data(a, a_data.data(), popnet_dtype_f32, 0, popnet_numel(a_shape)));
    REQUIRE(popnet_set_data(b, b_data.data(), popnet_dtype_f32, 0, popnet_numel(b_shape)));

    c->src[0] = a;
    c->src[1] = b;
    popnet_compute_matmul(c);

    std::vector<float> output_data(popnet_numel(c_shape));
    REQUIRE(popnet_get_data(c, output_data.data(), popnet_dtype_f32, 0, popnet_numel(c_shape)));
    for (size_t i = 0; i < output_data.size(); ++i)
    {
      CHECK(output_data[i] == doctest::Approx(expected_data[i]).epsilon(1e-5f).scale(1.0f));
    }
  }

  SUBCASE("[[1,2,3; 4,5,6], [7,8,9; 10,11,12]] x [[1,0,0; 0,1,0; 0,0,1]]^T - f32 (outer=2)")
  {
    popnet_shape a_shape = {{1, 2, 2, 3}}; // (outer=2, m=2, k=3)
    popnet_shape b_shape = {{1, 1, 3, 3}}; // identity
    popnet_shape c_shape = {{1, 2, 2, 3}}; // (outer=2, m=2, n=3)

    std::vector<float> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> b_data = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::vector<float> expected_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    auto ref_data = matmul_ref<popnet_dtype_f32>(a_data, b_data, 2, 2, 3, 3);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* a = popnet_empty(popnet_dtype_f32, a_shape);
    popnet_tensor* b = popnet_empty(popnet_dtype_f32, b_shape);
    popnet_tensor* c = popnet_empty(popnet_dtype_f32, c_shape);

    REQUIRE(popnet_set_data(a, a_data.data(), popnet_dtype_f32, 0, popnet_numel(a_shape)));
    REQUIRE(popnet_set_data(b, b_data.data(), popnet_dtype_f32, 0, popnet_numel(b_shape)));

    c->src[0] = a;
    c->src[1] = b;
    popnet_compute_matmul(c);

    std::vector<float> output_data(popnet_numel(c_shape));
    REQUIRE(popnet_get_data(c, output_data.data(), popnet_dtype_f32, 0, popnet_numel(c_shape)));
    for (size_t i = 0; i < output_data.size(); ++i)
    {
      CHECK(output_data[i] == doctest::Approx(expected_data[i]).epsilon(1e-5f).scale(1.0f));
    }
  }

  SUBCASE("[[1,2;3,4]] x [[5,6;7,8;9,10]]^T - f32")
  {
    popnet_shape a_shape = {{1, 1, 2, 2}}; // (m=2, k=2)
    popnet_shape b_shape = {{1, 1, 3, 2}}; // (n=3, k=2)
    popnet_shape c_shape = {{1, 1, 2, 3}}; // (m=2, n=3)

    std::vector<float> a_data = {1, 2, 3, 4};
    std::vector<float> b_data = {5, 6, 7, 8, 9, 10};
    std::vector<float> expected_data = {1 * 5 + 2 * 6, 1 * 7 + 2 * 8, 1 * 9 + 2 * 10,
                                        3 * 5 + 4 * 6, 3 * 7 + 4 * 8, 3 * 9 + 4 * 10};

    auto ref_data = matmul_ref<popnet_dtype_f32>(a_data, b_data, 1, 2, 3, 2);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* a = popnet_empty(popnet_dtype_f32, a_shape);
    popnet_tensor* b = popnet_empty(popnet_dtype_f32, b_shape);
    popnet_tensor* c = popnet_empty(popnet_dtype_f32, c_shape);

    REQUIRE(popnet_set_data(a, a_data.data(), popnet_dtype_f32, 0, popnet_numel(a_shape)));
    REQUIRE(popnet_set_data(b, b_data.data(), popnet_dtype_f32, 0, popnet_numel(b_shape)));

    c->src[0] = a;
    c->src[1] = b;
    popnet_compute_matmul(c);

    std::vector<float> output_data(popnet_numel(c_shape));
    REQUIRE(popnet_get_data(c, output_data.data(), popnet_dtype_f32, 0, popnet_numel(c_shape)));
    for (size_t i = 0; i < output_data.size(); ++i)
    {
      CHECK(output_data[i] == doctest::Approx(expected_data[i]).epsilon(1e-5f).scale(1.0f));
    }
  }

  auto run = []<popnet_dtype dtype, auto lower, auto upper>(popnet_shape a_shape, popnet_shape b_shape) {
    using dtype_traits = dtype_traits<dtype>;
    using scalar_t = typename dtype_traits::scalar_t;

    const size_t outer_size = a_shape.dim[0] * a_shape.dim[1];
    const size_t m = a_shape.dim[2];
    const size_t k = a_shape.dim[3];
    const size_t n = b_shape.dim[2];
    const popnet_shape c_shape = {{a_shape.dim[0], a_shape.dim[1], m, n}};

    const auto a_data = generate_scalar_data<dtype>(popnet_numel(a_shape), lower, upper, false);
    const auto b_data = generate_scalar_data<dtype>(popnet_numel(b_shape), lower, upper, false);

    const auto expected_data = matmul_ref<dtype>(a_data, b_data, outer_size, m, n, k);

    popnet_tensor* A = popnet_empty(dtype, a_shape);
    popnet_tensor* B = popnet_empty(dtype, b_shape);
    popnet_tensor* C = popnet_empty(dtype, c_shape);

    REQUIRE(popnet_set_data(A, a_data.data(), dtype, 0, popnet_numel(a_shape)));
    REQUIRE(popnet_set_data(B, b_data.data(), dtype, 0, popnet_numel(b_shape)));

    C->src[0] = A;
    C->src[1] = B;
    popnet_compute_matmul(C);

    std::vector<scalar_t> output_data(popnet_numel(c_shape));
    REQUIRE(popnet_get_data(C, output_data.data(), dtype, 0, popnet_numel(c_shape)));

    for (size_t i = 0; i < output_data.size(); ++i)
    {
      if constexpr (std::is_floating_point_v<scalar_t>)
      {
        CHECK(output_data[i] == doctest::Approx(expected_data[i]).epsilon(1e-4f).scale(1.0f));
      }
      else
      {
        CHECK(output_data[i] == expected_data[i]);
      }
    }
  };

  SUBCASE("Gen 1x1x1x1 x 1x1x1x1^T - f32")
  {
    run.operator()<popnet_dtype_f32, -10.0f, 10.0f>(popnet_shape{{1, 1, 1, 1}}, popnet_shape{{1, 1, 1, 1}});
  };
  SUBCASE("Gen 1x1x1x5 x 1x1x2x5^T - f32")
  {
    run.operator()<popnet_dtype_f32, -10.0f, 10.0f>(popnet_shape{{1, 1, 1, 5}}, popnet_shape{{1, 1, 2, 5}});
  };
  SUBCASE("Gen 1x1x2x3 x 1x1x4x3^T - f32")
  {
    run.operator()<popnet_dtype_f32, -10.0f, 10.0f>(popnet_shape{{1, 1, 2, 3}}, popnet_shape{{1, 1, 4, 3}});
  };
  SUBCASE("Gen 1x1x3x2 x 1x1x3x2^T - f32")
  {
    run.operator()<popnet_dtype_f32, -10.0f, 10.0f>(popnet_shape{{1, 1, 3, 2}}, popnet_shape{{1, 1, 3, 2}});
  };
  SUBCASE("Gen 1x1x3x3 x 1x1x3x3^T - f32")
  {
    run.operator()<popnet_dtype_f32, -10.0f, 10.0f>(popnet_shape{{1, 1, 3, 3}}, popnet_shape{{1, 1, 3, 3}});
  };
  SUBCASE("Gen 1x2x2x3 x 1x1x4x3^T - f32")
  {
    run.operator()<popnet_dtype_f32, -10.0f, 10.0f>(popnet_shape{{1, 2, 2, 3}}, popnet_shape{{1, 1, 4, 3}});
  };
  SUBCASE("Gen 2x2x3x3 x 1x1x3x3^T - f32")
  {
    run.operator()<popnet_dtype_f32, -10.0f, 10.0f>(popnet_shape{{2, 2, 3, 3}}, popnet_shape{{1, 1, 3, 3}});
  };
  SUBCASE("Gen 3x1x4x2 x 1x1x4x2^T - f32")
  {
    run.operator()<popnet_dtype_f32, -10.0f, 10.0f>(popnet_shape{{3, 1, 4, 2}}, popnet_shape{{1, 1, 4, 2}});
  };
  SUBCASE("Gen 4x2x16x16 x 1x1x32x16^T - f32")
  {
    run.operator()<popnet_dtype_f32, -10.0f, 10.0f>(popnet_shape{{4, 2, 16, 16}}, popnet_shape{{1, 1, 32, 16}});
  };
  SUBCASE("Gen 4x4x128x64 x 1x1x128x64^T - f32")
  {
    run.operator()<popnet_dtype_f32, -2.0f, 2.0f>(popnet_shape{{4, 4, 128, 64}}, popnet_shape{{1, 1, 128, 64}});
  };
  SUBCASE("Gen 1x1x512x512 x 1x1x512x512^T - f32")
  {
    run.operator()<popnet_dtype_f32, -1.0f, 1.0f>(popnet_shape{{1, 1, 512, 512}}, popnet_shape{{1, 1, 512, 512}});
  };
}

TEST_CASE("transpose")
{
  SUBCASE("[1,2,3; 4,5,6] - f32")
  {
    popnet_shape src_shape = {{1, 1, 2, 3}};
    popnet_shape dst_shape = popnet_transpose_shape(src_shape);

    std::vector<float> src_data = {{1, 2, 3, 4, 5, 6}};
    std::vector<float> expected_data = {{1, 4, 2, 5, 3, 6}};
    std::vector<float> ref_data = transpose_ref<popnet_dtype_f32>(src_data, 1, 2, 3);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* src = popnet_empty(popnet_dtype_f32, src_shape);
    popnet_tensor* dst = popnet_empty(popnet_dtype_f32, dst_shape);

    REQUIRE(popnet_set_data(src, src_data.data(), popnet_dtype_f32, 0, popnet_numel(src_shape)));
    dst->src[0] = src;

    popnet_compute_transpose(dst);

    std::vector<float> output_data(popnet_numel(dst_shape));
    REQUIRE(popnet_get_data(dst, output_data.data(), popnet_dtype_f32, 0, popnet_numel(dst_shape)));
    REQUIRE(output_data == expected_data);
  }

  SUBCASE("[1,2,3; 4,5,6; 7,8,9] - f32")
  {
    popnet_shape src_shape = {{1, 1, 3, 3}};
    popnet_shape dst_shape = popnet_transpose_shape(src_shape);

    std::vector<float> src_data = {{1, 2, 3, 4, 5, 6, 7, 8, 9}};
    std::vector<float> expected_data = {{1, 4, 7, 2, 5, 8, 3, 6, 9}};
    auto ref_data = transpose_ref<popnet_dtype_f32>(src_data, 1, 3, 3);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* src = popnet_empty(popnet_dtype_f32, src_shape);
    popnet_tensor* dst = popnet_empty(popnet_dtype_f32, dst_shape);

    REQUIRE(popnet_set_data(src, src_data.data(), popnet_dtype_f32, 0, popnet_numel(src_shape)));

    dst->src[0] = src;
    popnet_compute_transpose(dst);

    std::vector<float> output_data(popnet_numel(dst_shape));
    REQUIRE(popnet_get_data(dst, output_data.data(), popnet_dtype_f32, 0, popnet_numel(dst_shape)));
    REQUIRE(output_data == expected_data);
  }

  SUBCASE("[10,20,30,40] - f32")
  {
    popnet_shape src_shape = {{1, 1, 1, 4}};
    popnet_shape dst_shape = popnet_transpose_shape(src_shape);

    std::vector<float> src_data = {{10, 20, 30, 40}};
    std::vector<float> expected_data = {{10, 20, 30, 40}};
    auto ref_data = transpose_ref<popnet_dtype_f32>(src_data, 1, 1, 4);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* src = popnet_empty(popnet_dtype_f32, src_shape);
    popnet_tensor* dst = popnet_empty(popnet_dtype_f32, dst_shape);

    REQUIRE(popnet_set_data(src, src_data.data(), popnet_dtype_f32, 0, popnet_numel(src_shape)));

    dst->src[0] = src;
    popnet_compute_transpose(dst);

    std::vector<float> output_data(popnet_numel(dst_shape));
    REQUIRE(popnet_get_data(dst, output_data.data(), popnet_dtype_f32, 0, popnet_numel(dst_shape)));
    REQUIRE(output_data == expected_data);
  }

  SUBCASE("[1,-1; 2,-2; 3,-3; 4,-4] - f32")
  {
    popnet_shape src_shape = {{1, 1, 4, 2}};
    popnet_shape dst_shape = popnet_transpose_shape(src_shape);

    std::vector<float> src_data = {{1, -1, 2, -2, 3, -3, 4, -4}};
    std::vector<float> expected_data = {{1, 2, 3, 4, -1, -2, -3, -4}};
    auto ref_data = transpose_ref<popnet_dtype_f32>(src_data, 1, 4, 2);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* src = popnet_empty(popnet_dtype_f32, src_shape);
    popnet_tensor* dst = popnet_empty(popnet_dtype_f32, dst_shape);

    REQUIRE(popnet_set_data(src, src_data.data(), popnet_dtype_f32, 0, popnet_numel(src_shape)));

    dst->src[0] = src;
    popnet_compute_transpose(dst);

    std::vector<float> output_data(popnet_numel(dst_shape));
    REQUIRE(popnet_get_data(dst, output_data.data(), popnet_dtype_f32, 0, popnet_numel(dst_shape)));
    REQUIRE(output_data == expected_data);
  }

  SUBCASE("[1,2,3; 4,5,6; 7,8,9; 10,11,12; 13,14,15] - f32")
  {
    popnet_shape src_shape = {{1, 1, 5, 3}};
    popnet_shape dst_shape = popnet_transpose_shape(src_shape);

    std::vector<float> src_data = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    std::vector<float> expected_data = {{1, 4, 7, 10, 13, 2, 5, 8, 11, 14, 3, 6, 9, 12, 15}};
    auto ref_data = transpose_ref<popnet_dtype_f32>(src_data, 1, 5, 3);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* src = popnet_empty(popnet_dtype_f32, src_shape);
    popnet_tensor* dst = popnet_empty(popnet_dtype_f32, dst_shape);

    REQUIRE(popnet_set_data(src, src_data.data(), popnet_dtype_f32, 0, popnet_numel(src_shape)));

    dst->src[0] = src;
    popnet_compute_transpose(dst);

    std::vector<float> output_data(popnet_numel(dst_shape));
    REQUIRE(popnet_get_data(dst, output_data.data(), popnet_dtype_f32, 0, popnet_numel(dst_shape)));
    REQUIRE(output_data == expected_data);
  }

  SUBCASE("[[1,2,3; 4,5,6], [7,8,9; 10,11,12]] - f32 (outer=2)")
  {
    popnet_shape src_shape = {{1, 2, 2, 3}};
    popnet_shape dst_shape = popnet_transpose_shape(src_shape);

    std::vector<float> src_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> expected_data = {1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12};

    auto ref_data = transpose_ref<popnet_dtype_f32>(src_data, 2, 2, 3);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* src = popnet_empty(popnet_dtype_f32, src_shape);
    popnet_tensor* dst = popnet_empty(popnet_dtype_f32, dst_shape);

    REQUIRE(popnet_set_data(src, src_data.data(), popnet_dtype_f32, 0, popnet_numel(src_shape)));
    dst->src[0] = src;
    popnet_compute_transpose(dst);

    std::vector<float> output_data(popnet_numel(dst_shape));
    REQUIRE(popnet_get_data(dst, output_data.data(), popnet_dtype_f32, 0, popnet_numel(dst_shape)));
    REQUIRE(output_data == expected_data);
  }

  SUBCASE("[[1,2; 3,4], [5,6; 7,8], [9,10; 11,12]] - f32 (outer=3)")
  {
    popnet_shape src_shape = {{3, 1, 2, 2}};
    popnet_shape dst_shape = popnet_transpose_shape(src_shape);

    std::vector<float> src_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> expected_data = {1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12};

    auto ref_data = transpose_ref<popnet_dtype_f32>(src_data, 3, 2, 2);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* src = popnet_empty(popnet_dtype_f32, src_shape);
    popnet_tensor* dst = popnet_empty(popnet_dtype_f32, dst_shape);

    REQUIRE(popnet_set_data(src, src_data.data(), popnet_dtype_f32, 0, popnet_numel(src_shape)));
    dst->src[0] = src;
    popnet_compute_transpose(dst);

    std::vector<float> output_data(popnet_numel(dst_shape));
    REQUIRE(popnet_get_data(dst, output_data.data(), popnet_dtype_f32, 0, popnet_numel(dst_shape)));
    REQUIRE(output_data == expected_data);
  }

  SUBCASE("[[1,2,3; 4,5,6], [7,8,9; 10,11,12], [13,14,15; 16,17,18], [19,20,21; 22,23,24]] - f32")
  {
    popnet_shape src_shape = {{2, 2, 2, 3}};
    popnet_shape dst_shape = popnet_transpose_shape(src_shape);

    std::vector<float> src_data = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    std::vector<float> expected_data = {1,  4,  2,  5,  3,  6,  7,  10, 8,  11, 9,  12,
                                        13, 16, 14, 17, 15, 18, 19, 22, 20, 23, 21, 24};

    auto ref_data = transpose_ref<popnet_dtype_f32>(src_data, 4, 2, 3);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* src = popnet_empty(popnet_dtype_f32, src_shape);
    popnet_tensor* dst = popnet_empty(popnet_dtype_f32, dst_shape);

    REQUIRE(popnet_set_data(src, src_data.data(), popnet_dtype_f32, 0, popnet_numel(src_shape)));
    dst->src[0] = src;
    popnet_compute_transpose(dst);

    std::vector<float> output_data(popnet_numel(dst_shape));
    REQUIRE(popnet_get_data(dst, output_data.data(), popnet_dtype_f32, 0, popnet_numel(dst_shape)));
    REQUIRE(output_data == expected_data);
  }

  auto run = []<popnet_dtype dtype, auto lower, auto upper>(popnet_shape src_shape) {
    const size_t outer_size = src_shape.dim[0] * src_shape.dim[1];
    const popnet_shape dst_shape = popnet_transpose_shape(src_shape);

    const auto src_data = generate_scalar_data<dtype>(popnet_numel(src_shape), lower, upper, false);
    const auto expected_data = transpose_ref<dtype>(src_data, outer_size, src_shape.dim[2], src_shape.dim[3]);

    popnet_tensor* src = popnet_empty(popnet_dtype_f32, src_shape);
    popnet_tensor* dst = popnet_empty(popnet_dtype_f32, dst_shape);

    REQUIRE(popnet_set_data(src, src_data.data(), popnet_dtype_f32, 0, popnet_numel(src_shape)));

    dst->src[0] = src;
    popnet_compute_transpose(dst);

    std::vector<float> output_data(popnet_numel(dst_shape));
    REQUIRE(popnet_get_data(dst, output_data.data(), popnet_dtype_f32, 0, popnet_numel(dst_shape)));
    REQUIRE(output_data == expected_data);
  };

  SUBCASE("Gen 1x1x1x1 - f32") { run.operator()<popnet_dtype_f32, -100.0f, 100.0f>(popnet_shape{{1, 1, 1, 1}}); };
  SUBCASE("Gen 1x1x1x5 - f32") { run.operator()<popnet_dtype_f32, -100.0f, 100.0f>(popnet_shape{{1, 1, 1, 5}}); };
  SUBCASE("Gen 1x1x2x3 - f32") { run.operator()<popnet_dtype_f32, -100.0f, 100.0f>(popnet_shape{{1, 1, 2, 3}}); };
  SUBCASE("Gen 1x1x3x2 - f32") { run.operator()<popnet_dtype_f32, -100.0f, 100.0f>(popnet_shape{{1, 1, 3, 2}}); };
  SUBCASE("Gen 1x1x3x3 - f32") { run.operator()<popnet_dtype_f32, -100.0f, 100.0f>(popnet_shape{{1, 1, 3, 3}}); };
  SUBCASE("Gen 1x1x5x3 - f32") { run.operator()<popnet_dtype_f32, -100.0f, 100.0f>(popnet_shape{{1, 1, 5, 3}}); };
  SUBCASE("Gen 1x2x2x3 - f32") { run.operator()<popnet_dtype_f32, -100.0f, 100.0f>(popnet_shape{{1, 2, 2, 3}}); };
  SUBCASE("Gen 2x2x3x3 - f32") { run.operator()<popnet_dtype_f32, -100.0f, 100.0f>(popnet_shape{{2, 2, 3, 3}}); };
  SUBCASE("Gen 3x1x4x2 - f32") { run.operator()<popnet_dtype_f32, -100.0f, 100.0f>(popnet_shape{{3, 1, 4, 2}}); };
  SUBCASE("Gen 4x2x16x16 - f32") { run.operator()<popnet_dtype_f32, -100.0f, 100.0f>(popnet_shape{{4, 2, 16, 16}}); };
  SUBCASE("Gen 4x4x128x64 - f32") { run.operator()<popnet_dtype_f32, -100.0f, 100.0f>(popnet_shape{{4, 4, 128, 64}}); };
  SUBCASE("Gen 16x16x1000x1000 - f32")
  {
    run.operator()<popnet_dtype_f32, -100.0f, 100.0f>(popnet_shape{{1, 1, 1000, 1000}});
  };
  SUBCASE("Gen 16x16x500x1000 - f32")
  {
    run.operator()<popnet_dtype_f32, -100.0f, 100.0f>(popnet_shape{{1, 1, 500, 1000}});
  };
}

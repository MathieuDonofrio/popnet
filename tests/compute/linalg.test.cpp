#include "linalg.h"

#include <doctest/doctest.h>
#include <numeric>
#include <vector>

#include "ops.h"
#include "simd.h"
#include "tensor.h"
#include "test_utils.h"

template <popnet_type dtype> static auto transpose_ref(const auto& src, size_t outer_size, size_t n, size_t m)
{
  using traits = simd_traits<dtype>;
  using scalar_t = typename traits::scalar_t;

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

TEST_CASE("transpose")
{
  SUBCASE("[1,2,3; 4,5,6] - f32")
  {
    popnet_shape src_shape = {{1, 1, 2, 3}};
    popnet_shape dst_shape = popnet_transpose_shape(src_shape);

    std::vector<float> src_data = {{1, 2, 3, 4, 5, 6}};
    std::vector<float> expected_data = {{1, 4, 2, 5, 3, 6}};
    std::vector<float> ref_data = transpose_ref<POPNET_TYPE_F32>(src_data, 1, 2, 3);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* src = popnet_empty(POPNET_TYPE_F32, src_shape);
    popnet_tensor* dst = popnet_empty(POPNET_TYPE_F32, dst_shape);

    REQUIRE(popnet_set_data(src, src_data.data(), POPNET_TYPE_F32, 0, popnet_nelems(src_shape)));
    dst->src[0] = src;

    popnet_compute_transpose(dst);

    std::vector<float> output_data(popnet_nelems(dst_shape));
    REQUIRE(popnet_get_data(dst, output_data.data(), POPNET_TYPE_F32, 0, popnet_nelems(dst_shape)));
    REQUIRE(output_data == expected_data);
  }

  SUBCASE("[1,2,3; 4,5,6; 7,8,9] - f32")
  {
    popnet_shape src_shape = {{1, 1, 3, 3}};
    popnet_shape dst_shape = popnet_transpose_shape(src_shape);

    std::vector<float> src_data = {{1, 2, 3, 4, 5, 6, 7, 8, 9}};
    std::vector<float> expected_data = {{1, 4, 7, 2, 5, 8, 3, 6, 9}};
    auto ref_data = transpose_ref<POPNET_TYPE_F32>(src_data, 1, 3, 3);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* src = popnet_empty(POPNET_TYPE_F32, src_shape);
    popnet_tensor* dst = popnet_empty(POPNET_TYPE_F32, dst_shape);

    REQUIRE(popnet_set_data(src, src_data.data(), POPNET_TYPE_F32, 0, popnet_nelems(src_shape)));

    dst->src[0] = src;
    popnet_compute_transpose(dst);

    std::vector<float> output_data(popnet_nelems(dst_shape));
    REQUIRE(popnet_get_data(dst, output_data.data(), POPNET_TYPE_F32, 0, popnet_nelems(dst_shape)));
    REQUIRE(output_data == expected_data);
  }

  SUBCASE("[10,20,30,40] - f32")
  {
    popnet_shape src_shape = {{1, 1, 1, 4}};
    popnet_shape dst_shape = popnet_transpose_shape(src_shape);

    std::vector<float> src_data = {{10, 20, 30, 40}};
    std::vector<float> expected_data = {{10, 20, 30, 40}};
    auto ref_data = transpose_ref<POPNET_TYPE_F32>(src_data, 1, 1, 4);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* src = popnet_empty(POPNET_TYPE_F32, src_shape);
    popnet_tensor* dst = popnet_empty(POPNET_TYPE_F32, dst_shape);

    REQUIRE(popnet_set_data(src, src_data.data(), POPNET_TYPE_F32, 0, popnet_nelems(src_shape)));

    dst->src[0] = src;
    popnet_compute_transpose(dst);

    std::vector<float> output_data(popnet_nelems(dst_shape));
    REQUIRE(popnet_get_data(dst, output_data.data(), POPNET_TYPE_F32, 0, popnet_nelems(dst_shape)));
    REQUIRE(output_data == expected_data);
  }

  SUBCASE("[1,-1; 2,-2; 3,-3; 4,-4] - f32")
  {
    popnet_shape src_shape = {{1, 1, 4, 2}};
    popnet_shape dst_shape = popnet_transpose_shape(src_shape);

    std::vector<float> src_data = {{1, -1, 2, -2, 3, -3, 4, -4}};
    std::vector<float> expected_data = {{1, 2, 3, 4, -1, -2, -3, -4}};
    auto ref_data = transpose_ref<POPNET_TYPE_F32>(src_data, 1, 4, 2);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* src = popnet_empty(POPNET_TYPE_F32, src_shape);
    popnet_tensor* dst = popnet_empty(POPNET_TYPE_F32, dst_shape);

    REQUIRE(popnet_set_data(src, src_data.data(), POPNET_TYPE_F32, 0, popnet_nelems(src_shape)));

    dst->src[0] = src;
    popnet_compute_transpose(dst);

    std::vector<float> output_data(popnet_nelems(dst_shape));
    REQUIRE(popnet_get_data(dst, output_data.data(), POPNET_TYPE_F32, 0, popnet_nelems(dst_shape)));
    REQUIRE(output_data == expected_data);
  }

  SUBCASE("[1,2,3; 4,5,6; 7,8,9; 10,11,12; 13,14,15] - f32")
  {
    popnet_shape src_shape = {{1, 1, 5, 3}};
    popnet_shape dst_shape = popnet_transpose_shape(src_shape);

    std::vector<float> src_data = {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    std::vector<float> expected_data = {{1, 4, 7, 10, 13, 2, 5, 8, 11, 14, 3, 6, 9, 12, 15}};
    auto ref_data = transpose_ref<POPNET_TYPE_F32>(src_data, 1, 5, 3);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* src = popnet_empty(POPNET_TYPE_F32, src_shape);
    popnet_tensor* dst = popnet_empty(POPNET_TYPE_F32, dst_shape);

    REQUIRE(popnet_set_data(src, src_data.data(), POPNET_TYPE_F32, 0, popnet_nelems(src_shape)));

    dst->src[0] = src;
    popnet_compute_transpose(dst);

    std::vector<float> output_data(popnet_nelems(dst_shape));
    REQUIRE(popnet_get_data(dst, output_data.data(), POPNET_TYPE_F32, 0, popnet_nelems(dst_shape)));
    REQUIRE(output_data == expected_data);
  }

  SUBCASE("[[1,2,3; 4,5,6], [7,8,9; 10,11,12]] - f32 (outer=2)")
  {
    popnet_shape src_shape = {{1, 2, 2, 3}};
    popnet_shape dst_shape = popnet_transpose_shape(src_shape);

    std::vector<float> src_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> expected_data = {1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12};

    auto ref_data = transpose_ref<POPNET_TYPE_F32>(src_data, 2, 2, 3);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* src = popnet_empty(POPNET_TYPE_F32, src_shape);
    popnet_tensor* dst = popnet_empty(POPNET_TYPE_F32, dst_shape);

    REQUIRE(popnet_set_data(src, src_data.data(), POPNET_TYPE_F32, 0, popnet_nelems(src_shape)));
    dst->src[0] = src;
    popnet_compute_transpose(dst);

    std::vector<float> output_data(popnet_nelems(dst_shape));
    REQUIRE(popnet_get_data(dst, output_data.data(), POPNET_TYPE_F32, 0, popnet_nelems(dst_shape)));
    REQUIRE(output_data == expected_data);
  }

  SUBCASE("[[1,2; 3,4], [5,6; 7,8], [9,10; 11,12]] - f32 (outer=3)")
  {
    popnet_shape src_shape = {{3, 1, 2, 2}};
    popnet_shape dst_shape = popnet_transpose_shape(src_shape);

    std::vector<float> src_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> expected_data = {1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12};

    auto ref_data = transpose_ref<POPNET_TYPE_F32>(src_data, 3, 2, 2);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* src = popnet_empty(POPNET_TYPE_F32, src_shape);
    popnet_tensor* dst = popnet_empty(POPNET_TYPE_F32, dst_shape);

    REQUIRE(popnet_set_data(src, src_data.data(), POPNET_TYPE_F32, 0, popnet_nelems(src_shape)));
    dst->src[0] = src;
    popnet_compute_transpose(dst);

    std::vector<float> output_data(popnet_nelems(dst_shape));
    REQUIRE(popnet_get_data(dst, output_data.data(), POPNET_TYPE_F32, 0, popnet_nelems(dst_shape)));
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

    auto ref_data = transpose_ref<POPNET_TYPE_F32>(src_data, 4, 2, 3);
    REQUIRE(ref_data == expected_data);

    popnet_tensor* src = popnet_empty(POPNET_TYPE_F32, src_shape);
    popnet_tensor* dst = popnet_empty(POPNET_TYPE_F32, dst_shape);

    REQUIRE(popnet_set_data(src, src_data.data(), POPNET_TYPE_F32, 0, popnet_nelems(src_shape)));
    dst->src[0] = src;
    popnet_compute_transpose(dst);

    std::vector<float> output_data(popnet_nelems(dst_shape));
    REQUIRE(popnet_get_data(dst, output_data.data(), POPNET_TYPE_F32, 0, popnet_nelems(dst_shape)));
    REQUIRE(output_data == expected_data);
  }

  auto run = []<popnet_type dtype, auto lower, auto upper>(popnet_shape src_shape) {
    const size_t outer_size = src_shape.dim[0] * src_shape.dim[1];
    const popnet_shape dst_shape = popnet_transpose_shape(src_shape);

    const auto src_data = generate_scalar_data<dtype>(popnet_nelems(src_shape), lower, upper, false);
    const auto expected_data = transpose_ref<dtype>(src_data, outer_size, src_shape.dim[2], src_shape.dim[3]);

    popnet_tensor* src = popnet_empty(POPNET_TYPE_F32, src_shape);
    popnet_tensor* dst = popnet_empty(POPNET_TYPE_F32, dst_shape);

    REQUIRE(popnet_set_data(src, src_data.data(), POPNET_TYPE_F32, 0, popnet_nelems(src_shape)));

    dst->src[0] = src;
    popnet_compute_transpose(dst);

    std::vector<float> output_data(popnet_nelems(dst_shape));
    REQUIRE(popnet_get_data(dst, output_data.data(), POPNET_TYPE_F32, 0, popnet_nelems(dst_shape)));
    REQUIRE(output_data == expected_data);
  };

  SUBCASE("Gen 1x1x1x1 - f32") { run.operator()<POPNET_TYPE_F32, -100.0f, 100.0f>(popnet_shape{{1, 1, 1, 1}}); };
  SUBCASE("Gen 1x1x1x5 - f32") { run.operator()<POPNET_TYPE_F32, -100.0f, 100.0f>(popnet_shape{{1, 1, 1, 5}}); };
  SUBCASE("Gen 1x1x2x3 - f32") { run.operator()<POPNET_TYPE_F32, -100.0f, 100.0f>(popnet_shape{{1, 1, 2, 3}}); };
  SUBCASE("Gen 1x1x3x2 - f32") { run.operator()<POPNET_TYPE_F32, -100.0f, 100.0f>(popnet_shape{{1, 1, 3, 2}}); };
  SUBCASE("Gen 1x1x3x3 - f32") { run.operator()<POPNET_TYPE_F32, -100.0f, 100.0f>(popnet_shape{{1, 1, 3, 3}}); };
  SUBCASE("Gen 1x1x5x3 - f32") { run.operator()<POPNET_TYPE_F32, -100.0f, 100.0f>(popnet_shape{{1, 1, 5, 3}}); };
  SUBCASE("Gen 1x2x2x3 - f32") { run.operator()<POPNET_TYPE_F32, -100.0f, 100.0f>(popnet_shape{{1, 2, 2, 3}}); };
  SUBCASE("Gen 2x2x3x3 - f32") { run.operator()<POPNET_TYPE_F32, -100.0f, 100.0f>(popnet_shape{{2, 2, 3, 3}}); };
  SUBCASE("Gen 3x1x4x2 - f32") { run.operator()<POPNET_TYPE_F32, -100.0f, 100.0f>(popnet_shape{{3, 1, 4, 2}}); };
  SUBCASE("Gen 4x2x16x16 - f32") { run.operator()<POPNET_TYPE_F32, -100.0f, 100.0f>(popnet_shape{{4, 2, 16, 16}}); };
  SUBCASE("Gen 4x4x128x64 - f32") { run.operator()<POPNET_TYPE_F32, -100.0f, 100.0f>(popnet_shape{{4, 4, 128, 64}}); };
  SUBCASE("Gen 16x16x1000x1000 - f32")
  {
    run.operator()<POPNET_TYPE_F32, -100.0f, 100.0f>(popnet_shape{{1, 1, 1000, 1000}});
  };
  SUBCASE("Gen 16x16x500x1000 - f32")
  {
    run.operator()<POPNET_TYPE_F32, -100.0f, 100.0f>(popnet_shape{{1, 1, 500, 1000}});
  };
}

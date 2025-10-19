#include "ternarize.h"

#include "vector_types.h"

#include <bit>
#include <immintrin.h>
#include <intrin.h>

POPNET_INLINE void ternarize(const vf32* POPNET_RESTRICT a, vtrit* POPNET_RESTRICT b, size_t n) noexcept
{
  constexpr size_t VF32_PER_VTERN = vtrit::elements / vf32::elements;
  const size_t blocks = (n + vf32::elements - 1) / vf32::elements;

  for (size_t i = 0; i < blocks; i += VF32_PER_VTERN)
  {
    struct
    {
      __mmask16 sgn[VF32_PER_VTERN] = {};
      __mmask16 mag[VF32_PER_VTERN] = {};
    } masks;

    for (size_t j = 0; j < VF32_PER_VTERN && (i + j) < blocks; j++)
    {
      const auto [aa] = vf32::load(a + i + j);
      const __m512i vround = _mm512_cvt_roundps_epi32(aa, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      const __m512i vternary = _mm512_min_epi32(_mm512_max_epi32(vround, _mm512_set1_epi32(-1)), _mm512_set1_epi32(1));
      masks.sgn[j] = _mm512_movepi32_mask(vternary);
      masks.mag[j] = _mm512_cmpneq_epi32_mask(vternary, _mm512_setzero_si512());
    }

    std::bit_cast<vtrit>(masks).store(b + i / VF32_PER_VTERN);
  }
}

// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/unipass-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/dwconv.h"


void xnn_f32_dwconv_minmax_ukernel_25p4c__neonfma_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const float32x4_t vmax = vdupq_n_f32(params->scalar.max);
  const float32x4_t vmin = vdupq_n_f32(params->scalar.min);
  // Supertile chosen to be ~L1 cache size for sizeof(**input) + sizeof(*weights) + sizeof(*output).
  for (size_t co = 0; co < channels; co += 84) {
    size_t channel_supertile = 84;
    if (co + channel_supertile > channels) {
      channel_supertile = channels - co;
    }
    float* o = output + co;
    const float** input_co = input;
    size_t ow = output_width;
    // output_increment assumes we process `channels` at a time. Fix it.
    size_t oi = output_increment + (channels - channel_supertile) * sizeof(float);
    do {
      const float* i0 = input_co[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 += co;
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input_co[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 += co;
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input_co[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 += co;
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input_co[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 += co;
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input_co[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 += co;
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      const float* i5 = input_co[5];
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 += co;
        i5 = (const float*) ((uintptr_t) i5 + input_offset);
      }
      const float* i6 = input_co[6];
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 += co;
        i6 = (const float*) ((uintptr_t) i6 + input_offset);
      }
      const float* i7 = input_co[7];
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 += co;
        i7 = (const float*) ((uintptr_t) i7 + input_offset);
      }
      const float* i8 = input_co[8];
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 += co;
        i8 = (const float*) ((uintptr_t) i8 + input_offset);
      }
      const float* i9 = input_co[9];
      assert(i9 != NULL);
      if XNN_UNPREDICTABLE(i9 != zero) {
        i9 += co;
        i9 = (const float*) ((uintptr_t) i9 + input_offset);
      }
      const float* i10 = input_co[10];
      assert(i10 != NULL);
      if XNN_UNPREDICTABLE(i10 != zero) {
        i10 += co;
        i10 = (const float*) ((uintptr_t) i10 + input_offset);
      }
      const float* i11 = input_co[11];
      assert(i11 != NULL);
      if XNN_UNPREDICTABLE(i11 != zero) {
        i11 += co;
        i11 = (const float*) ((uintptr_t) i11 + input_offset);
      }
      const float* i12 = input_co[12];
      assert(i12 != NULL);
      if XNN_UNPREDICTABLE(i12 != zero) {
        i12 += co;
        i12 = (const float*) ((uintptr_t) i12 + input_offset);
      }
      const float* i13 = input_co[13];
      assert(i13 != NULL);
      if XNN_UNPREDICTABLE(i13 != zero) {
        i13 += co;
        i13 = (const float*) ((uintptr_t) i13 + input_offset);
      }
      const float* i14 = input_co[14];
      assert(i14 != NULL);
      if XNN_UNPREDICTABLE(i14 != zero) {
        i14 += co;
        i14 = (const float*) ((uintptr_t) i14 + input_offset);
      }
      const float* i15 = input_co[15];
      assert(i15 != NULL);
      if XNN_UNPREDICTABLE(i15 != zero) {
        i15 += co;
        i15 = (const float*) ((uintptr_t) i15 + input_offset);
      }
      const float* i16 = input_co[16];
      assert(i16 != NULL);
      if XNN_UNPREDICTABLE(i16 != zero) {
        i16 += co;
        i16 = (const float*) ((uintptr_t) i16 + input_offset);
      }
      const float* i17 = input_co[17];
      assert(i17 != NULL);
      if XNN_UNPREDICTABLE(i17 != zero) {
        i17 += co;
        i17 = (const float*) ((uintptr_t) i17 + input_offset);
      }
      const float* i18 = input_co[18];
      assert(i18 != NULL);
      if XNN_UNPREDICTABLE(i18 != zero) {
        i18 += co;
        i18 = (const float*) ((uintptr_t) i18 + input_offset);
      }
      const float* i19 = input_co[19];
      assert(i19 != NULL);
      if XNN_UNPREDICTABLE(i19 != zero) {
        i19 += co;
        i19 = (const float*) ((uintptr_t) i19 + input_offset);
      }
      const float* i20 = input_co[20];
      assert(i20 != NULL);
      if XNN_UNPREDICTABLE(i20 != zero) {
        i20 += co;
        i20 = (const float*) ((uintptr_t) i20 + input_offset);
      }
      const float* i21 = input_co[21];
      assert(i21 != NULL);
      if XNN_UNPREDICTABLE(i21 != zero) {
        i21 += co;
        i21 = (const float*) ((uintptr_t) i21 + input_offset);
      }
      const float* i22 = input_co[22];
      assert(i22 != NULL);
      if XNN_UNPREDICTABLE(i22 != zero) {
        i22 += co;
        i22 = (const float*) ((uintptr_t) i22 + input_offset);
      }
      const float* i23 = input_co[23];
      assert(i23 != NULL);
      if XNN_UNPREDICTABLE(i23 != zero) {
        i23 += co;
        i23 = (const float*) ((uintptr_t) i23 + input_offset);
      }
      const float* i24 = input_co[24];
      assert(i24 != NULL);
      if XNN_UNPREDICTABLE(i24 != zero) {
        i24 += co;
        i24 = (const float*) ((uintptr_t) i24 + input_offset);
      }

      input_co = (const float**) ((uintptr_t) input_co + input_stride);

      size_t c = channel_supertile;
      const float* w = weights + co * 26;
      for (; c >= 4; c -= 4) {
        float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;


        const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

        const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
        float32x4_t vacc0123p1 = vmulq_f32(vi1x0123, vk1x0123);

        const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);

        const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi3x0123, vk3x0123);

        const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;
        const float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);

        const float32x4_t vi5x0123 = vld1q_f32(i5); i5 += 4;
        const float32x4_t vk5x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi5x0123, vk5x0123);

        const float32x4_t vi6x0123 = vld1q_f32(i6); i6 += 4;
        const float32x4_t vk6x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi6x0123, vk6x0123);

        const float32x4_t vi7x0123 = vld1q_f32(i7); i7 += 4;
        const float32x4_t vk7x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi7x0123, vk7x0123);

        const float32x4_t vi8x0123 = vld1q_f32(i8); i8 += 4;
        const float32x4_t vk8x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi8x0123, vk8x0123);

        const float32x4_t vi9x0123 = vld1q_f32(i9); i9 += 4;
        const float32x4_t vk9x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi9x0123, vk9x0123);

        const float32x4_t vi10x0123 = vld1q_f32(i10); i10 += 4;
        const float32x4_t vk10x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi10x0123, vk10x0123);

        const float32x4_t vi11x0123 = vld1q_f32(i11); i11 += 4;
        const float32x4_t vk11x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi11x0123, vk11x0123);

        const float32x4_t vi12x0123 = vld1q_f32(i12); i12 += 4;
        const float32x4_t vk12x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi12x0123, vk12x0123);

        const float32x4_t vi13x0123 = vld1q_f32(i13); i13 += 4;
        const float32x4_t vk13x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi13x0123, vk13x0123);

        const float32x4_t vi14x0123 = vld1q_f32(i14); i14 += 4;
        const float32x4_t vk14x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi14x0123, vk14x0123);

        const float32x4_t vi15x0123 = vld1q_f32(i15); i15 += 4;
        const float32x4_t vk15x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi15x0123, vk15x0123);

        const float32x4_t vi16x0123 = vld1q_f32(i16); i16 += 4;
        const float32x4_t vk16x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi16x0123, vk16x0123);

        const float32x4_t vi17x0123 = vld1q_f32(i17); i17 += 4;
        const float32x4_t vk17x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi17x0123, vk17x0123);

        const float32x4_t vi18x0123 = vld1q_f32(i18); i18 += 4;
        const float32x4_t vk18x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi18x0123, vk18x0123);

        const float32x4_t vi19x0123 = vld1q_f32(i19); i19 += 4;
        const float32x4_t vk19x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi19x0123, vk19x0123);

        const float32x4_t vi20x0123 = vld1q_f32(i20); i20 += 4;
        const float32x4_t vk20x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi20x0123, vk20x0123);

        const float32x4_t vi21x0123 = vld1q_f32(i21); i21 += 4;
        const float32x4_t vk21x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi21x0123, vk21x0123);

        const float32x4_t vi22x0123 = vld1q_f32(i22); i22 += 4;
        const float32x4_t vk22x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi22x0123, vk22x0123);

        const float32x4_t vi23x0123 = vld1q_f32(i23); i23 += 4;
        const float32x4_t vk23x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi23x0123, vk23x0123);

        const float32x4_t vi24x0123 = vld1q_f32(i24); i24 += 4;
        const float32x4_t vk24x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi24x0123, vk24x0123);

        // Add up all accumulators to vacc0123p0
        vacc0123p0 = vaddq_f32(vacc0123p0, vacc0123p1);

        float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
        vacc0123 = vminq_f32(vacc0123, vmax);

        vst1q_f32(o, vacc0123); o += 4;
      }
      if XNN_UNLIKELY(c != 0) {
        float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;


        const float32x4_t vi0x0123 = vld1q_f32(i0);
        const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

        const float32x4_t vi1x0123 = vld1q_f32(i1);
        const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
        float32x4_t vacc0123p1 = vmulq_f32(vi1x0123, vk1x0123);

        const float32x4_t vi2x0123 = vld1q_f32(i2);
        const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);

        const float32x4_t vi3x0123 = vld1q_f32(i3);
        const float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi3x0123, vk3x0123);

        const float32x4_t vi4x0123 = vld1q_f32(i4);
        const float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);

        const float32x4_t vi5x0123 = vld1q_f32(i5);
        const float32x4_t vk5x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi5x0123, vk5x0123);

        const float32x4_t vi6x0123 = vld1q_f32(i6);
        const float32x4_t vk6x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi6x0123, vk6x0123);

        const float32x4_t vi7x0123 = vld1q_f32(i7);
        const float32x4_t vk7x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi7x0123, vk7x0123);

        const float32x4_t vi8x0123 = vld1q_f32(i8);
        const float32x4_t vk8x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi8x0123, vk8x0123);

        const float32x4_t vi9x0123 = vld1q_f32(i9);
        const float32x4_t vk9x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi9x0123, vk9x0123);

        const float32x4_t vi10x0123 = vld1q_f32(i10);
        const float32x4_t vk10x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi10x0123, vk10x0123);

        const float32x4_t vi11x0123 = vld1q_f32(i11);
        const float32x4_t vk11x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi11x0123, vk11x0123);

        const float32x4_t vi12x0123 = vld1q_f32(i12);
        const float32x4_t vk12x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi12x0123, vk12x0123);

        const float32x4_t vi13x0123 = vld1q_f32(i13);
        const float32x4_t vk13x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi13x0123, vk13x0123);

        const float32x4_t vi14x0123 = vld1q_f32(i14);
        const float32x4_t vk14x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi14x0123, vk14x0123);

        const float32x4_t vi15x0123 = vld1q_f32(i15);
        const float32x4_t vk15x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi15x0123, vk15x0123);

        const float32x4_t vi16x0123 = vld1q_f32(i16);
        const float32x4_t vk16x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi16x0123, vk16x0123);

        const float32x4_t vi17x0123 = vld1q_f32(i17);
        const float32x4_t vk17x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi17x0123, vk17x0123);

        const float32x4_t vi18x0123 = vld1q_f32(i18);
        const float32x4_t vk18x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi18x0123, vk18x0123);

        const float32x4_t vi19x0123 = vld1q_f32(i19);
        const float32x4_t vk19x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi19x0123, vk19x0123);

        const float32x4_t vi20x0123 = vld1q_f32(i20);
        const float32x4_t vk20x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi20x0123, vk20x0123);

        const float32x4_t vi21x0123 = vld1q_f32(i21);
        const float32x4_t vk21x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi21x0123, vk21x0123);

        const float32x4_t vi22x0123 = vld1q_f32(i22);
        const float32x4_t vk22x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi22x0123, vk22x0123);

        const float32x4_t vi23x0123 = vld1q_f32(i23);
        const float32x4_t vk23x0123 = vld1q_f32(w); w += 4;
        vacc0123p1 = vfmaq_f32(vacc0123p1, vi23x0123, vk23x0123);

        const float32x4_t vi24x0123 = vld1q_f32(i24);
        const float32x4_t vk24x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi24x0123, vk24x0123);

        // Add up all accumulators to vacc0123p0
        vacc0123p0 = vaddq_f32(vacc0123p0, vacc0123p1);

        float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
        vacc0123 = vminq_f32(vacc0123, vmax);

        float32x2_t vacc01 = vget_low_f32(vacc0123);
        if (c & 2) {
          vst1_f32(o, vacc01); o += 2;
          vacc01 = vget_high_f32(vacc0123);
        }
        if (c & 1) {
          vst1_lane_f32(o, vacc01, 0); o += 1;
        }
      }

      o = (float*) ((uintptr_t) o + oi);
    } while (--ow != 0);
  }
}

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


void xnn_f32_dwconv_minmax_ukernel_9p4c__neon(
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
  for (size_t co = 0; co < channels; co += 228) {
    size_t channel_supertile = 228;
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

      input_co = (const float**) ((uintptr_t) input_co + input_stride);

      size_t c = channel_supertile;
      const float* w = weights + co * 10;
      for (; c >= 4; c -= 4) {
        float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;


        const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi0x0123, vk0x0123);

        const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi1x0123, vk1x0123);

        const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi2x0123, vk2x0123);

        const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi3x0123, vk3x0123);

        const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;
        const float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi4x0123, vk4x0123);

        const float32x4_t vi5x0123 = vld1q_f32(i5); i5 += 4;
        const float32x4_t vk5x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi5x0123, vk5x0123);

        const float32x4_t vi6x0123 = vld1q_f32(i6); i6 += 4;
        const float32x4_t vk6x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi6x0123, vk6x0123);

        const float32x4_t vi7x0123 = vld1q_f32(i7); i7 += 4;
        const float32x4_t vk7x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi7x0123, vk7x0123);

        const float32x4_t vi8x0123 = vld1q_f32(i8); i8 += 4;
        const float32x4_t vk8x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi8x0123, vk8x0123);


        float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
        vacc0123 = vminq_f32(vacc0123, vmax);

        vst1q_f32(o, vacc0123); o += 4;
      }
      if XNN_UNLIKELY(c != 0) {
        float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;


        const float32x4_t vi0x0123 = vld1q_f32(i0);
        const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi0x0123, vk0x0123);

        const float32x4_t vi1x0123 = vld1q_f32(i1);
        const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi1x0123, vk1x0123);

        const float32x4_t vi2x0123 = vld1q_f32(i2);
        const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi2x0123, vk2x0123);

        const float32x4_t vi3x0123 = vld1q_f32(i3);
        const float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi3x0123, vk3x0123);

        const float32x4_t vi4x0123 = vld1q_f32(i4);
        const float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi4x0123, vk4x0123);

        const float32x4_t vi5x0123 = vld1q_f32(i5);
        const float32x4_t vk5x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi5x0123, vk5x0123);

        const float32x4_t vi6x0123 = vld1q_f32(i6);
        const float32x4_t vk6x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi6x0123, vk6x0123);

        const float32x4_t vi7x0123 = vld1q_f32(i7);
        const float32x4_t vk7x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi7x0123, vk7x0123);

        const float32x4_t vi8x0123 = vld1q_f32(i8);
        const float32x4_t vk8x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi8x0123, vk8x0123);


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

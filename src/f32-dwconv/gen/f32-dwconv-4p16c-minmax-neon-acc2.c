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


void xnn_f32_dwconv_minmax_ukernel_4p16c__neon_acc2(
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
  for (size_t co = 0; co < channels; co += 512) {
    size_t channel_supertile = 512;
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

      input_co = (const float**) ((uintptr_t) input_co + input_stride);

      size_t c = channel_supertile;
      const float* w = weights + co * 5;
      for (; c >= 16; c -= 16) {
        float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;
        float32x4_t vacc4567p0 = vld1q_f32(w); w += 4;
        float32x4_t vacc89ABp0 = vld1q_f32(w); w += 4;
        float32x4_t vaccCDEFp0 = vld1q_f32(w); w += 4;


        const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi0x4567 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi0x89AB = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi0xCDEF = vld1q_f32(i0); i0 += 4;
        const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk0x4567 = vld1q_f32(w); w += 4;
        const float32x4_t vk0x89AB = vld1q_f32(w); w += 4;
        const float32x4_t vk0xCDEF = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi0x0123, vk0x0123);
        vacc4567p0 = vmlaq_f32(vacc4567p0, vi0x4567, vk0x4567);
        vacc89ABp0 = vmlaq_f32(vacc89ABp0, vi0x89AB, vk0x89AB);
        vaccCDEFp0 = vmlaq_f32(vaccCDEFp0, vi0xCDEF, vk0xCDEF);

        const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi1x4567 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi1x89AB = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi1xCDEF = vld1q_f32(i1); i1 += 4;
        const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk1x4567 = vld1q_f32(w); w += 4;
        const float32x4_t vk1x89AB = vld1q_f32(w); w += 4;
        const float32x4_t vk1xCDEF = vld1q_f32(w); w += 4;
        float32x4_t vacc0123p1 = vmulq_f32(vi1x0123, vk1x0123);
        float32x4_t vacc4567p1 = vmulq_f32(vi1x4567, vk1x4567);
        float32x4_t vacc89ABp1 = vmulq_f32(vi1x89AB, vk1x89AB);
        float32x4_t vaccCDEFp1 = vmulq_f32(vi1xCDEF, vk1xCDEF);

        const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi2x4567 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi2x89AB = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi2xCDEF = vld1q_f32(i2); i2 += 4;
        const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk2x4567 = vld1q_f32(w); w += 4;
        const float32x4_t vk2x89AB = vld1q_f32(w); w += 4;
        const float32x4_t vk2xCDEF = vld1q_f32(w); w += 4;
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi2x0123, vk2x0123);
        vacc4567p0 = vmlaq_f32(vacc4567p0, vi2x4567, vk2x4567);
        vacc89ABp0 = vmlaq_f32(vacc89ABp0, vi2x89AB, vk2x89AB);
        vaccCDEFp0 = vmlaq_f32(vaccCDEFp0, vi2xCDEF, vk2xCDEF);

        const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi3x4567 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi3x89AB = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi3xCDEF = vld1q_f32(i3); i3 += 4;
        const float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk3x4567 = vld1q_f32(w); w += 4;
        const float32x4_t vk3x89AB = vld1q_f32(w); w += 4;
        const float32x4_t vk3xCDEF = vld1q_f32(w); w += 4;
        vacc0123p1 = vmlaq_f32(vacc0123p1, vi3x0123, vk3x0123);
        vacc4567p1 = vmlaq_f32(vacc4567p1, vi3x4567, vk3x4567);
        vacc89ABp1 = vmlaq_f32(vacc89ABp1, vi3x89AB, vk3x89AB);
        vaccCDEFp1 = vmlaq_f32(vaccCDEFp1, vi3xCDEF, vk3xCDEF);

        // Add up all accumulators to vacc0123456789ABCDEFp0
        vacc0123p0 = vaddq_f32(vacc0123p0, vacc0123p1);
        vacc4567p0 = vaddq_f32(vacc4567p0, vacc4567p1);
        vacc89ABp0 = vaddq_f32(vacc89ABp0, vacc89ABp1);
        vaccCDEFp0 = vaddq_f32(vaccCDEFp0, vaccCDEFp1);

        float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
        float32x4_t vacc4567 = vmaxq_f32(vacc4567p0, vmin);
        float32x4_t vacc89AB = vmaxq_f32(vacc89ABp0, vmin);
        float32x4_t vaccCDEF = vmaxq_f32(vaccCDEFp0, vmin);
        vacc0123 = vminq_f32(vacc0123, vmax);
        vacc4567 = vminq_f32(vacc4567, vmax);
        vacc89AB = vminq_f32(vacc89AB, vmax);
        vaccCDEF = vminq_f32(vaccCDEF, vmax);

        vst1q_f32(o, vacc0123); o += 4;
        vst1q_f32(o, vacc4567); o += 4;
        vst1q_f32(o, vacc89AB); o += 4;
        vst1q_f32(o, vaccCDEF); o += 4;
      }
      for (; c >= 4; c -= 4) {
        float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;


        const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vk0x0123 = vld1q_f32(w + 12);
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi0x0123, vk0x0123);

        const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vk1x0123 = vld1q_f32(w + 28);
        float32x4_t vacc0123p1 = vmulq_f32(vi1x0123, vk1x0123);

        const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vk2x0123 = vld1q_f32(w + 44);
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi2x0123, vk2x0123);

        const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vk3x0123 = vld1q_f32(w + 60);
        vacc0123p1 = vmlaq_f32(vacc0123p1, vi3x0123, vk3x0123);

        // Add up all accumulators to vacc0123p0
        vacc0123p0 = vaddq_f32(vacc0123p0, vacc0123p1);

        float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
        vacc0123 = vminq_f32(vacc0123, vmax);

        vst1q_f32(o, vacc0123); o += 4;
      }
      if XNN_UNLIKELY(c != 0) {
        float32x4_t vacc0123p0 = vld1q_f32(w);


        const float32x4_t vi0x0123 = vld1q_f32(i0);
        const float32x4_t vk0x0123 = vld1q_f32(w + 16);
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi0x0123, vk0x0123);

        const float32x4_t vi1x0123 = vld1q_f32(i1);
        const float32x4_t vk1x0123 = vld1q_f32(w + 32);
        float32x4_t vacc0123p1 = vmulq_f32(vi1x0123, vk1x0123);

        const float32x4_t vi2x0123 = vld1q_f32(i2);
        const float32x4_t vk2x0123 = vld1q_f32(w + 48);
        vacc0123p0 = vmlaq_f32(vacc0123p0, vi2x0123, vk2x0123);

        const float32x4_t vi3x0123 = vld1q_f32(i3);
        const float32x4_t vk3x0123 = vld1q_f32(w + 64);
        vacc0123p1 = vmlaq_f32(vacc0123p1, vi3x0123, vk3x0123);

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

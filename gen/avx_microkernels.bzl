"""
Microkernel filenames lists for avx.

Auto-generated file. Do not edit!
  Generator: tools/update-microkernels.py
"""

PROD_AVX_MICROKERNEL_SRCS = [
    "src/f16-f32-vcvt/gen/f16-f32-vcvt-avx-int16-u16.c",
    "src/f32-dwconv/gen/f32-dwconv-3p16c-minmax-avx.c",
    "src/f32-dwconv/gen/f32-dwconv-4p16c-minmax-avx.c",
    "src/f32-dwconv/gen/f32-dwconv-9p16c-minmax-avx.c",
    "src/f32-dwconv/gen/f32-dwconv-25p16c-minmax-avx.c",
    "src/f32-f16-vcvt/gen/f32-f16-vcvt-avx-u24.c",
    "src/f32-gemm/gen/f32-gemm-1x8-minmax-avx-broadcast.c",
    "src/f32-gemm/gen/f32-gemm-1x16-minmax-avx-broadcast.c",
    "src/f32-gemm/gen/f32-gemm-5x8-minmax-avx-broadcast.c",
    "src/f32-gemm/gen/f32-gemm-5x16-minmax-avx-broadcast.c",
    "src/f32-igemm/gen/f32-igemm-1x8-minmax-avx-broadcast.c",
    "src/f32-igemm/gen/f32-igemm-1x16-minmax-avx-broadcast.c",
    "src/f32-igemm/gen/f32-igemm-5x8-minmax-avx-broadcast.c",
    "src/f32-igemm/gen/f32-igemm-5x16-minmax-avx-broadcast.c",
    "src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x16-minmax-avx-broadcast.c",
    "src/f32-qc4w-gemm/gen/f32-qc4w-gemm-3x16-minmax-avx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x16-minmax-avx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-5x16-minmax-avx-broadcast.c",
    "src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx-u32.c",
    "src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx-u32.c",
    "src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx-c32.c",
    "src/f32-rminmax/gen/f32-rmax-avx-u32-acc4.c",
    "src/f32-rminmax/gen/f32-rminmax-avx-u32-acc4.c",
    "src/f32-rsum/gen/f32-rsum-avx-u32-acc4.c",
    "src/f32-vapproxgelu/gen/f32-vapproxgelu-avx-rational-12-10-div.c",
    "src/f32-vbinary/gen/f32-vadd-avx-u16.c",
    "src/f32-vbinary/gen/f32-vaddc-avx-u16.c",
    "src/f32-vbinary/gen/f32-vdiv-avx-u16.c",
    "src/f32-vbinary/gen/f32-vdivc-avx-u16.c",
    "src/f32-vbinary/gen/f32-vmax-avx-u16.c",
    "src/f32-vbinary/gen/f32-vmaxc-avx-u16.c",
    "src/f32-vbinary/gen/f32-vmin-avx-u16.c",
    "src/f32-vbinary/gen/f32-vminc-avx-u16.c",
    "src/f32-vbinary/gen/f32-vmul-avx-u16.c",
    "src/f32-vbinary/gen/f32-vmulc-avx-u16.c",
    "src/f32-vbinary/gen/f32-vprelu-avx-u16.c",
    "src/f32-vbinary/gen/f32-vpreluc-avx-u16.c",
    "src/f32-vbinary/gen/f32-vrdivc-avx-u16.c",
    "src/f32-vbinary/gen/f32-vrpreluc-avx-u16.c",
    "src/f32-vbinary/gen/f32-vrsubc-avx-u16.c",
    "src/f32-vbinary/gen/f32-vsqrdiff-avx-u16.c",
    "src/f32-vbinary/gen/f32-vsqrdiffc-avx-u16.c",
    "src/f32-vbinary/gen/f32-vsub-avx-u16.c",
    "src/f32-vbinary/gen/f32-vsubc-avx-u16.c",
    "src/f32-vclamp/gen/f32-vclamp-avx-u16.c",
    "src/f32-vcopysign/gen/f32-vcopysign-avx.c",
    "src/f32-vcopysign/gen/f32-vcopysignc-avx.c",
    "src/f32-vcopysign/gen/f32-vrcopysignc-avx.c",
    "src/f32-velu/gen/f32-velu-avx-rr2-lut4-p4-perm-u32.c",
    "src/f32-vgelu/gen/f32-vgelu-avx-rational-12-10-div.c",
    "src/f32-vhswish/gen/f32-vhswish-avx-u16.c",
    "src/f32-vlrelu/gen/f32-vlrelu-avx-u16.c",
    "src/f32-vrnd/gen/f32-vrndd-avx-u16.c",
    "src/f32-vrnd/gen/f32-vrndne-avx-u16.c",
    "src/f32-vrnd/gen/f32-vrndu-avx-u16.c",
    "src/f32-vrnd/gen/f32-vrndz-avx-u16.c",
    "src/f32-vrsqrt/gen/f32-vrsqrt-avx-rsqrt-u16.c",
    "src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-nr2-u16.c",
    "src/f32-vsqrt/gen/f32-vsqrt-avx-rsqrt-u16.c",
    "src/f32-vtanh/gen/f32-vtanh-avx-rational-9-8-div.c",
    "src/f32-vunary/gen/f32-vabs-avx.c",
    "src/f32-vunary/gen/f32-vneg-avx.c",
    "src/f32-vunary/gen/f32-vsqr-avx.c",
    "src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x4c8-minmax-avx-ld128.c",
    "src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x4c8-minmax-avx-ld128.c",
    "src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-avx-mul16-add16.c",
    "src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-avx-mul16-add16.c",
    "src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx-u32.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p16c-minmax-fp32-avx-mul16-add16.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-avx-mul16-add16.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-avx-mul16-add16.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c8-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c8-minmax-fp32-avx-ld128.c",
    "src/qs8-vadd/gen/qs8-vadd-minmax-avx-mul32-ld32-u8.c",
    "src/qs8-vaddc/gen/qs8-vaddc-minmax-avx-mul32-ld32-u8.c",
    "src/qs8-vcvt/gen/qs8-vcvt-avx-u32.c",
    "src/qs8-vlrelu/gen/qs8-vlrelu-avx-u32.c",
    "src/qs8-vmul/gen/qs8-vmul-minmax-fp32-avx-mul16-ld64-u16.c",
    "src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-avx-mul16-ld64-u16.c",
    "src/qu8-dwconv/gen/qu8-dwconv-9p16c-minmax-fp32-avx-mul16.c",
    "src/qu8-dwconv/gen/qu8-dwconv-25p16c-minmax-fp32-avx-mul16.c",
    "src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx-u32.c",
    "src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-avx-ld128.c",
    "src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-avx-ld128.c",
    "src/qu8-igemm/gen/qu8-igemm-1x4c8-minmax-fp32-avx-ld128.c",
    "src/qu8-igemm/gen/qu8-igemm-2x4c8-minmax-fp32-avx-ld128.c",
    "src/qu8-vadd/gen/qu8-vadd-minmax-avx-mul32-ld32-u8.c",
    "src/qu8-vaddc/gen/qu8-vaddc-minmax-avx-mul32-ld32-u8.c",
    "src/qu8-vcvt/gen/qu8-vcvt-avx-u32.c",
    "src/qu8-vlrelu/gen/qu8-vlrelu-avx-u32.c",
    "src/qu8-vmul/gen/qu8-vmul-minmax-fp32-avx-mul16-ld64-u16.c",
    "src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-avx-mul16-ld64-u16.c",
    "src/x8-lut/gen/x8-lut-avx-u64.c",
    "src/x32-packw/gen/x32-packw-x8-gemm-gio-avx-u8.c",
    "src/x32-packw/gen/x32-packw-x8-gemm-goi-avx-u4.c",
    "src/x32-packw/gen/x32-packw-x16-gemm-gio-avx-u8.c",
    "src/x32-packw/gen/x32-packw-x16-gemm-goi-avx-u4.c",
    "src/x32-packw/gen/x32-packw-x16s4-gemm-goi-avx-u4.c",
    "src/x32-transposec/gen/x32-transposec-8x8-reuse-multi-avx.c",
    "src/x64-transposec/gen/x64-transposec-4x4-reuse-multi-avx.c",
]

NON_PROD_AVX_MICROKERNEL_SRCS = [
    "src/f16-f32-vcvt/gen/f16-f32-vcvt-avx-int16-u8.c",
    "src/f16-f32-vcvt/gen/f16-f32-vcvt-avx-int16-u24.c",
    "src/f16-f32-vcvt/gen/f16-f32-vcvt-avx-int16-u32.c",
    "src/f16-f32-vcvt/gen/f16-f32-vcvt-avx-int32-u8.c",
    "src/f16-f32-vcvt/gen/f16-f32-vcvt-avx-int32-u16.c",
    "src/f16-f32-vcvt/gen/f16-f32-vcvt-avx-int32-u24.c",
    "src/f16-f32-vcvt/gen/f16-f32-vcvt-avx-int32-u32.c",
    "src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-avx-acc2.c",
    "src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-avx.c",
    "src/f32-dwconv/gen/f32-dwconv-3p16c-minmax-avx-acc2.c",
    "src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-avx-acc2.c",
    "src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-avx.c",
    "src/f32-dwconv/gen/f32-dwconv-4p16c-minmax-avx-acc2.c",
    "src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-avx-acc2.c",
    "src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-avx.c",
    "src/f32-dwconv/gen/f32-dwconv-9p16c-minmax-avx-acc2.c",
    "src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-avx-acc2.c",
    "src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-avx.c",
    "src/f32-dwconv/gen/f32-dwconv-25p16c-minmax-avx-acc2.c",
    "src/f32-f16-vcvt/gen/f32-f16-vcvt-avx-u8.c",
    "src/f32-f16-vcvt/gen/f32-f16-vcvt-avx-u16.c",
    "src/f32-f16-vcvt/gen/f32-f16-vcvt-avx-u32.c",
    "src/f32-gemm/gen/f32-gemm-3x16-minmax-avx-broadcast.c",
    "src/f32-gemm/gen/f32-gemm-4x8-minmax-avx-broadcast.c",
    "src/f32-gemm/gen/f32-gemm-4x16-minmax-avx-broadcast.c",
    "src/f32-gemm/gen/f32-gemm-6x8-minmax-avx-broadcast.c",
    "src/f32-gemm/gen/f32-gemm-6x16-minmax-avx-broadcast.c",
    "src/f32-gemm/gen/f32-gemm-7x8-minmax-avx-broadcast.c",
    "src/f32-gemminc/gen/f32-gemminc-1x8-minmax-avx-broadcast.c",
    "src/f32-gemminc/gen/f32-gemminc-1x16-minmax-avx-broadcast.c",
    "src/f32-gemminc/gen/f32-gemminc-3x16-minmax-avx-broadcast.c",
    "src/f32-gemminc/gen/f32-gemminc-4x8-minmax-avx-broadcast.c",
    "src/f32-gemminc/gen/f32-gemminc-4x16-minmax-avx-broadcast.c",
    "src/f32-gemminc/gen/f32-gemminc-5x8-minmax-avx-broadcast.c",
    "src/f32-gemminc/gen/f32-gemminc-5x16-minmax-avx-broadcast.c",
    "src/f32-gemminc/gen/f32-gemminc-6x8-minmax-avx-broadcast.c",
    "src/f32-gemminc/gen/f32-gemminc-6x16-minmax-avx-broadcast.c",
    "src/f32-gemminc/gen/f32-gemminc-7x8-minmax-avx-broadcast.c",
    "src/f32-igemm/gen/f32-igemm-3x16-minmax-avx-broadcast.c",
    "src/f32-igemm/gen/f32-igemm-4x8-minmax-avx-broadcast.c",
    "src/f32-igemm/gen/f32-igemm-4x16-minmax-avx-broadcast.c",
    "src/f32-igemm/gen/f32-igemm-6x8-minmax-avx-broadcast.c",
    "src/f32-igemm/gen/f32-igemm-6x16-minmax-avx-broadcast.c",
    "src/f32-igemm/gen/f32-igemm-7x8-minmax-avx-broadcast.c",
    "src/f32-prelu/gen/f32-prelu-avx-2x16.c",
    "src/f32-qc4w-gemm/gen/f32-qc4w-gemm-2x16-minmax-avx-broadcast.c",
    "src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x16-minmax-avx-broadcast.c",
    "src/f32-qc4w-gemm/gen/f32-qc4w-gemm-5x16-minmax-avx-broadcast.c",
    "src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x16-minmax-avx-broadcast.c",
    "src/f32-qc4w-gemm/gen/f32-qc4w-gemm-7x16-minmax-avx-broadcast.c",
    "src/f32-qc4w-gemm/gen/f32-qc4w-gemm-8x16-minmax-avx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-2x16-minmax-avx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-3x16-minmax-avx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x16-minmax-avx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x16-minmax-avx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-7x16-minmax-avx-broadcast.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-8x16-minmax-avx-broadcast.c",
    "src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx-u8.c",
    "src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx-u16.c",
    "src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx-u24.c",
    "src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx-u8.c",
    "src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx-u16.c",
    "src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx-u24.c",
    "src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx-c16.c",
    "src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx-c64.c",
    "src/f32-rminmax/gen/f32-rmax-avx-u8.c",
    "src/f32-rminmax/gen/f32-rmax-avx-u16-acc2.c",
    "src/f32-rminmax/gen/f32-rmax-avx-u24-acc3.c",
    "src/f32-rminmax/gen/f32-rmax-avx-u32-acc2.c",
    "src/f32-rminmax/gen/f32-rmin-avx-u8.c",
    "src/f32-rminmax/gen/f32-rmin-avx-u16-acc2.c",
    "src/f32-rminmax/gen/f32-rmin-avx-u24-acc3.c",
    "src/f32-rminmax/gen/f32-rmin-avx-u32-acc2.c",
    "src/f32-rminmax/gen/f32-rmin-avx-u32-acc4.c",
    "src/f32-rminmax/gen/f32-rminmax-avx-u8.c",
    "src/f32-rminmax/gen/f32-rminmax-avx-u16-acc2.c",
    "src/f32-rminmax/gen/f32-rminmax-avx-u24-acc3.c",
    "src/f32-rminmax/gen/f32-rminmax-avx-u32-acc2.c",
    "src/f32-rsum/gen/f32-rsum-avx-u8.c",
    "src/f32-rsum/gen/f32-rsum-avx-u16-acc2.c",
    "src/f32-rsum/gen/f32-rsum-avx-u24-acc3.c",
    "src/f32-rsum/gen/f32-rsum-avx-u32-acc2.c",
    "src/f32-vbinary/gen/f32-vadd-avx-u8.c",
    "src/f32-vbinary/gen/f32-vaddc-avx-u8.c",
    "src/f32-vbinary/gen/f32-vdiv-avx-u8.c",
    "src/f32-vbinary/gen/f32-vdivc-avx-u8.c",
    "src/f32-vbinary/gen/f32-vmax-avx-u8.c",
    "src/f32-vbinary/gen/f32-vmaxc-avx-u8.c",
    "src/f32-vbinary/gen/f32-vmin-avx-u8.c",
    "src/f32-vbinary/gen/f32-vminc-avx-u8.c",
    "src/f32-vbinary/gen/f32-vmul-avx-u8.c",
    "src/f32-vbinary/gen/f32-vmulc-avx-u8.c",
    "src/f32-vbinary/gen/f32-vprelu-avx-u8.c",
    "src/f32-vbinary/gen/f32-vpreluc-avx-u8.c",
    "src/f32-vbinary/gen/f32-vrdivc-avx-u8.c",
    "src/f32-vbinary/gen/f32-vrpreluc-avx-u8.c",
    "src/f32-vbinary/gen/f32-vrsubc-avx-u8.c",
    "src/f32-vbinary/gen/f32-vsqrdiff-avx-u8.c",
    "src/f32-vbinary/gen/f32-vsqrdiffc-avx-u8.c",
    "src/f32-vbinary/gen/f32-vsub-avx-u8.c",
    "src/f32-vbinary/gen/f32-vsubc-avx-u8.c",
    "src/f32-vclamp/gen/f32-vclamp-avx-u8.c",
    "src/f32-velu/gen/f32-velu-avx-rr2-lut4-p4-perm-u8.c",
    "src/f32-velu/gen/f32-velu-avx-rr2-lut4-p4-perm-u16.c",
    "src/f32-velu/gen/f32-velu-avx-rr2-lut4-p4-perm-u24.c",
    "src/f32-velu/gen/f32-velu-avx-rr2-lut16-p3-u8.c",
    "src/f32-velu/gen/f32-velu-avx-rr2-lut16-p3-u16.c",
    "src/f32-velu/gen/f32-velu-avx-rr2-lut16-p3-u24.c",
    "src/f32-velu/gen/f32-velu-avx-rr2-lut16-p3-u32.c",
    "src/f32-velu/gen/f32-velu-avx-rr2-p6-u8.c",
    "src/f32-velu/gen/f32-velu-avx-rr2-p6-u16.c",
    "src/f32-velu/gen/f32-velu-avx-rr2-p6-u24.c",
    "src/f32-velu/gen/f32-velu-avx-rr2-p6-u32.c",
    "src/f32-vhswish/gen/f32-vhswish-avx-u8.c",
    "src/f32-vlrelu/gen/f32-vlrelu-avx-u8.c",
    "src/f32-vrelu/gen/f32-vrelu-avx-u8.c",
    "src/f32-vrelu/gen/f32-vrelu-avx-u16.c",
    "src/f32-vrnd/gen/f32-vrndd-avx-u8.c",
    "src/f32-vrnd/gen/f32-vrndne-avx-u8.c",
    "src/f32-vrnd/gen/f32-vrndu-avx-u8.c",
    "src/f32-vrnd/gen/f32-vrndz-avx-u8.c",
    "src/f32-vrsqrt/gen/f32-vrsqrt-avx-rsqrt-u8.c",
    "src/f32-vrsqrt/gen/f32-vrsqrt-avx-rsqrt-u32.c",
    "src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-div-u8.c",
    "src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-div-u16.c",
    "src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-div-u24.c",
    "src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-div-u32.c",
    "src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-nr2-u8.c",
    "src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-nr2-u24.c",
    "src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-nr2-u32.c",
    "src/f32-vsqrt/gen/f32-vsqrt-avx-rsqrt-u8.c",
    "src/f32-vsqrt/gen/f32-vsqrt-avx-rsqrt-u32.c",
    "src/f32-vsqrt/gen/f32-vsqrt-avx-sqrt-u8.c",
    "src/f32-vsqrt/gen/f32-vsqrt-avx-sqrt-u16.c",
    "src/f32-vsqrt/gen/f32-vsqrt-avx-sqrt-u32.c",
    "src/f32-vtanh/gen/f32-vtanh-avx-rational-9-8-nr.c",
    "src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x4c8-minmax-avx-ld64.c",
    "src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x4c8-minmax-avx-ld64.c",
    "src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x4c8-minmax-avx-ld128.c",
    "src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x4c8-minmax-avx-ld64.c",
    "src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x4c8-minmax-avx-ld128.c",
    "src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x4c8-minmax-avx-ld64.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-avx-ld64.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-avx-ld128.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-avx-ld64.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-avx-ld128.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-avx-ld64.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-avx-ld128.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-avx-ld64.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-avx-ld128.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-avx-ld64.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-avx-ld128.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-avx-ld64.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-avx-ld128.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-avx-ld64.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-avx-ld128.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-avx-ld64.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-avx-ld128.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x4c8-minmax-avx-ld64.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x4c8-minmax-avx-ld128.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x4c8-minmax-avx-ld64.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x4c8-minmax-avx-ld128.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x4c8-minmax-avx-ld64.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x4c8-minmax-avx-ld128.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x4c8-minmax-avx-ld64.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x4c8-minmax-avx-ld128.c",
    "src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-avx-mul16-add16.c",
    "src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-avx-mul16.c",
    "src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-avx-mul32.c",
    "src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-avx-mul16.c",
    "src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-avx-mul32.c",
    "src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-avx-mul16-add16.c",
    "src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-avx-mul16.c",
    "src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-avx-mul32.c",
    "src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-avx-mul16.c",
    "src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-avx-mul32.c",
    "src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx-u8.c",
    "src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx-u16.c",
    "src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx-u24.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-avx-mul16-add16.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-avx-mul16.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-avx-mul32.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-avx-mul16.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-avx-mul32.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-avx-mul16-add16.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-avx-mul16.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-avx-mul32.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-avx-mul16.c",
    "src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-avx-mul32.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2s4-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2s4-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c8-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2s4-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2s4-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c8-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2s4-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2s4-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c8-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c8-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2-minmax-fp32-avx-ld128.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2s4-minmax-fp32-avx-ld64.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2s4-minmax-fp32-avx-ld128.c",
    "src/qs8-vadd/gen/qs8-vadd-minmax-avx-mul16-ld64-u8.c",
    "src/qs8-vadd/gen/qs8-vadd-minmax-avx-mul16-ld64-u16.c",
    "src/qs8-vadd/gen/qs8-vadd-minmax-avx-mul16-ld64-u24.c",
    "src/qs8-vadd/gen/qs8-vadd-minmax-avx-mul16-ld64-u32.c",
    "src/qs8-vadd/gen/qs8-vadd-minmax-avx-mul32-ld32-u16.c",
    "src/qs8-vadd/gen/qs8-vadd-minmax-avx-mul32-ld32-u24.c",
    "src/qs8-vadd/gen/qs8-vadd-minmax-avx-mul32-ld32-u32.c",
    "src/qs8-vaddc/gen/qs8-vaddc-minmax-avx-mul16-ld64-u8.c",
    "src/qs8-vaddc/gen/qs8-vaddc-minmax-avx-mul16-ld64-u16.c",
    "src/qs8-vaddc/gen/qs8-vaddc-minmax-avx-mul16-ld64-u24.c",
    "src/qs8-vaddc/gen/qs8-vaddc-minmax-avx-mul16-ld64-u32.c",
    "src/qs8-vaddc/gen/qs8-vaddc-minmax-avx-mul32-ld32-u16.c",
    "src/qs8-vaddc/gen/qs8-vaddc-minmax-avx-mul32-ld32-u24.c",
    "src/qs8-vaddc/gen/qs8-vaddc-minmax-avx-mul32-ld32-u32.c",
    "src/qs8-vcvt/gen/qs8-vcvt-avx-u8.c",
    "src/qs8-vcvt/gen/qs8-vcvt-avx-u16.c",
    "src/qs8-vlrelu/gen/qs8-vlrelu-avx-u8.c",
    "src/qs8-vlrelu/gen/qs8-vlrelu-avx-u16.c",
    "src/qs8-vmul/gen/qs8-vmul-minmax-fp32-avx-mul16-ld64-u8.c",
    "src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-avx-mul16-ld64-u8.c",
    "src/qu8-dwconv/gen/qu8-dwconv-9p8c-minmax-fp32-avx-mul16.c",
    "src/qu8-dwconv/gen/qu8-dwconv-9p8c-minmax-fp32-avx-mul32.c",
    "src/qu8-dwconv/gen/qu8-dwconv-9p16c-minmax-fp32-avx-mul32.c",
    "src/qu8-dwconv/gen/qu8-dwconv-25p8c-minmax-fp32-avx-mul16.c",
    "src/qu8-dwconv/gen/qu8-dwconv-25p8c-minmax-fp32-avx-mul32.c",
    "src/qu8-dwconv/gen/qu8-dwconv-25p16c-minmax-fp32-avx-mul32.c",
    "src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx-u8.c",
    "src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx-u16.c",
    "src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx-u24.c",
    "src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-avx-ld64.c",
    "src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-avx-ld128.c",
    "src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-avx-ld64.c",
    "src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-avx-ld128.c",
    "src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-avx-ld64.c",
    "src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-avx-ld64.c",
    "src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-avx-ld128.c",
    "src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-avx-ld64.c",
    "src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-avx-ld128.c",
    "src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-avx-ld64.c",
    "src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-avx-ld64.c",
    "src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-avx-ld128.c",
    "src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-avx-ld64.c",
    "src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-avx-ld128.c",
    "src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-avx-ld64.c",
    "src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-avx-ld128.c",
    "src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-avx-ld64.c",
    "src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-avx-ld128.c",
    "src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-avx-ld64.c",
    "src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-avx-ld128.c",
    "src/qu8-igemm/gen/qu8-igemm-1x4c2-minmax-fp32-avx-ld64.c",
    "src/qu8-igemm/gen/qu8-igemm-1x4c2-minmax-fp32-avx-ld128.c",
    "src/qu8-igemm/gen/qu8-igemm-1x4c2s4-minmax-fp32-avx-ld64.c",
    "src/qu8-igemm/gen/qu8-igemm-1x4c2s4-minmax-fp32-avx-ld128.c",
    "src/qu8-igemm/gen/qu8-igemm-1x4c8-minmax-fp32-avx-ld64.c",
    "src/qu8-igemm/gen/qu8-igemm-2x4c2-minmax-fp32-avx-ld64.c",
    "src/qu8-igemm/gen/qu8-igemm-2x4c2-minmax-fp32-avx-ld128.c",
    "src/qu8-igemm/gen/qu8-igemm-2x4c2s4-minmax-fp32-avx-ld64.c",
    "src/qu8-igemm/gen/qu8-igemm-2x4c2s4-minmax-fp32-avx-ld128.c",
    "src/qu8-igemm/gen/qu8-igemm-2x4c8-minmax-fp32-avx-ld64.c",
    "src/qu8-igemm/gen/qu8-igemm-3x4c2-minmax-fp32-avx-ld64.c",
    "src/qu8-igemm/gen/qu8-igemm-3x4c2-minmax-fp32-avx-ld128.c",
    "src/qu8-igemm/gen/qu8-igemm-3x4c2s4-minmax-fp32-avx-ld64.c",
    "src/qu8-igemm/gen/qu8-igemm-3x4c2s4-minmax-fp32-avx-ld128.c",
    "src/qu8-igemm/gen/qu8-igemm-3x4c8-minmax-fp32-avx-ld64.c",
    "src/qu8-igemm/gen/qu8-igemm-3x4c8-minmax-fp32-avx-ld128.c",
    "src/qu8-igemm/gen/qu8-igemm-4x4c2-minmax-fp32-avx-ld64.c",
    "src/qu8-igemm/gen/qu8-igemm-4x4c2-minmax-fp32-avx-ld128.c",
    "src/qu8-igemm/gen/qu8-igemm-4x4c2s4-minmax-fp32-avx-ld64.c",
    "src/qu8-igemm/gen/qu8-igemm-4x4c2s4-minmax-fp32-avx-ld128.c",
    "src/qu8-vadd/gen/qu8-vadd-minmax-avx-mul16-ld64-u8.c",
    "src/qu8-vadd/gen/qu8-vadd-minmax-avx-mul16-ld64-u16.c",
    "src/qu8-vadd/gen/qu8-vadd-minmax-avx-mul32-ld32-u16.c",
    "src/qu8-vaddc/gen/qu8-vaddc-minmax-avx-mul16-ld64-u8.c",
    "src/qu8-vaddc/gen/qu8-vaddc-minmax-avx-mul16-ld64-u16.c",
    "src/qu8-vaddc/gen/qu8-vaddc-minmax-avx-mul32-ld32-u16.c",
    "src/qu8-vcvt/gen/qu8-vcvt-avx-u8.c",
    "src/qu8-vcvt/gen/qu8-vcvt-avx-u16.c",
    "src/qu8-vlrelu/gen/qu8-vlrelu-avx-u8.c",
    "src/qu8-vlrelu/gen/qu8-vlrelu-avx-u16.c",
    "src/qu8-vmul/gen/qu8-vmul-minmax-fp32-avx-mul16-ld64-u8.c",
    "src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-avx-mul16-ld64-u8.c",
    "src/x8-lut/gen/x8-lut-avx-u16.c",
    "src/x8-lut/gen/x8-lut-avx-u32.c",
    "src/x8-lut/gen/x8-lut-avx-u48.c",
    "src/x32-packw/gen/x32-packw-x8-gemm-gio-avx-u1-prfm.c",
    "src/x32-packw/gen/x32-packw-x8-gemm-gio-avx-u1.c",
    "src/x32-packw/gen/x32-packw-x8-gemm-gio-avx-u8-prfm.c",
    "src/x32-packw/gen/x32-packw-x8-gemm-goi-avx-u4-prfm.c",
    "src/x32-packw/gen/x32-packw-x8s4-gemm-goi-avx-u4-prfm.c",
    "src/x32-packw/gen/x32-packw-x8s4-gemm-goi-avx-u4.c",
    "src/x32-packw/gen/x32-packw-x16-gemm-gio-avx-u1-prfm.c",
    "src/x32-packw/gen/x32-packw-x16-gemm-gio-avx-u1.c",
    "src/x32-packw/gen/x32-packw-x16-gemm-gio-avx-u8-prfm.c",
    "src/x32-packw/gen/x32-packw-x16-gemm-goi-avx-u4-prfm.c",
    "src/x32-packw/gen/x32-packw-x16s4-gemm-goi-avx-u4-prfm.c",
    "src/x32-packw/gen/x32-packw-x32-gemm-gio-avx-u1-prfm.c",
    "src/x32-packw/gen/x32-packw-x32-gemm-gio-avx-u1.c",
    "src/x32-packw/gen/x32-packw-x32-gemm-gio-avx-u8-prfm.c",
    "src/x32-packw/gen/x32-packw-x32-gemm-gio-avx-u8.c",
    "src/x32-transposec/gen/x32-transposec-8x8-multi-mov-avx.c",
    "src/x32-transposec/gen/x32-transposec-8x8-multi-switch-avx.c",
    "src/x32-transposec/gen/x32-transposec-8x8-reuse-mov-avx.c",
    "src/x32-transposec/gen/x32-transposec-8x8-reuse-switch-avx.c",
    "src/x64-transposec/gen/x64-transposec-4x4-multi-mov-avx.c",
    "src/x64-transposec/gen/x64-transposec-4x4-multi-multi-avx.c",
    "src/x64-transposec/gen/x64-transposec-4x4-multi-switch-avx.c",
    "src/x64-transposec/gen/x64-transposec-4x4-reuse-mov-avx.c",
    "src/x64-transposec/gen/x64-transposec-4x4-reuse-switch-avx.c",
]

ALL_AVX_MICROKERNEL_SRCS = PROD_AVX_MICROKERNEL_SRCS + NON_PROD_AVX_MICROKERNEL_SRCS

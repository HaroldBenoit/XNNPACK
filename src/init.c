// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <unistd.h>
#endif

#ifdef _MSC_VER
  #include <intrin.h>
#endif

#include "include/xnnpack.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/params.h"

#if XNN_ENABLE_KLEIDIAI
#include <stdio.h>

#include "kai/kai_common.h"
#endif  // XNN_ENABLE_KLEIDIAI

XNN_INIT_ONCE_GUARD(allocator);

static const struct xnn_allocator* volatile init_allocator = NULL;

static void init_allocator_config(void) {
  uint32_t init_flags = XNN_INIT_FLAG_XNNPACK;
  memcpy(&xnn_params.allocator, init_allocator, sizeof(struct xnn_allocator));
  xnn_params.init_flags = init_flags;
}

enum xnn_status xnn_initialize(const struct xnn_allocator* allocator) {
  const struct xnn_hardware_config* hardware_config = xnn_init_hardware_config();
  if (hardware_config == NULL) {
    xnn_log_error("XNNPACK initialization failed: hardware not supported");
    return xnn_status_unsupported_hardware;
  }

  if (allocator == NULL) {
    allocator = &xnn_default_allocator;
  }
  #ifdef _MSC_VER
    _InterlockedCompareExchangePointer((PVOID volatile*) &init_allocator, (PVOID) allocator, NULL);
  #else
    __sync_bool_compare_and_swap(&init_allocator, NULL, allocator);
  #endif
  XNN_INIT_ONCE(allocator);
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) != 0) {
    return xnn_status_success;
  } else {
    return xnn_status_unsupported_hardware;
  }

#if XNN_ENABLE_KLEIDIAI
  /* If we're using KleidiAI, log their version string. */
  static bool first = true;
  if (first) {
    first = false;
    fprintf(stderr,
            "XNNPACK built with KleidiAI v%s (see "
            "https://gitlab.arm.com/kleidi/kleidiai).\n",
            kai_get_version());
  }
#endif  // XNN_ENABLE_KLEIDIAI
}

enum xnn_status xnn_deinitialize(void) {
  return xnn_status_success;
}

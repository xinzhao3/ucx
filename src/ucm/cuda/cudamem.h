/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_CUDAMEM_H_
#define UCM_CUDAMEM_H_

#include <ucm/api/ucm.h>
#include <cuda.h>
#include <cuda_runtime.h>

ucs_status_t ucm_cudamem_install(int events);

void ucm_cudamem_event_test_callback(ucm_event_type_t event_type,
                                  ucm_event_t *event, void *arg);


cudaError_t ucm_override_cudaFree(void *addr);

#endif

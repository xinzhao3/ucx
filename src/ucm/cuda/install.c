/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cudamem.h"

#include <ucm/api/ucm.h>
#include <ucm/event/event.h>
#include <ucm/util/log.h>
#include <ucm/util/reloc.h>
#include <ucm/util/ucm_config.h>
#include <ucs/sys/math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <pthread.h>



static ucm_reloc_patch_t ucm_cudamem_symbol_patches[] = {
    {"cudaFree",   ucm_override_cudaFree},
    {NULL, NULL}
};

ucs_status_t ucm_cudamem_install()
{
    static int ucm_cudamem_installed = 0;
    static pthread_mutex_t install_mutex = PTHREAD_MUTEX_INITIALIZER;
    ucm_reloc_patch_t *patch;
    ucs_status_t status = UCS_OK;

    if (!ucm_global_config.enable_cuda_hooks) {
        ucm_debug("installing cudamem relocations is disabled by configuration");
        return UCS_ERR_UNSUPPORTED;
    }
    if (ucm_cudamem_installed) {
        return UCS_OK;
    }

    pthread_mutex_lock(&install_mutex);

    for (patch = ucm_cudamem_symbol_patches; patch->symbol != NULL; ++patch) {
        status = ucm_reloc_modify(patch);
        if (status != UCS_OK) {
            ucm_warn("failed to install relocation table entry for '%s'",
                     patch->symbol);
            goto out_unlock;
        }
    }
    ucm_cudamem_installed = 1;

out_unlock:
    pthread_mutex_unlock(&install_mutex);
    return status;
}

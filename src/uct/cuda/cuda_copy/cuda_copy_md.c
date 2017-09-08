/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "cuda_copy_md.h"

#include <string.h>
#include <limits.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>
#include <cuda_runtime.h>
#include <cuda.h>


static ucs_status_t uct_cuda_copy_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags         = UCT_MD_FLAG_REG | UCT_MD_FLAG_ADDR_DN;
    md_attr->cap.addr_dn_mask  = UCT_MD_ADDR_DOMAIN_CUDA;
    md_attr->cap.max_alloc     = 0;
    md_attr->cap.max_reg       = ULONG_MAX;
    md_attr->rkey_packed_size  = 0;
    md_attr->reg_cost.overhead = 0;
    md_attr->reg_cost.growth   = 0;
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_mkey_pack(uct_md_h md, uct_mem_h memh,
                                      void *rkey_buffer)
{
    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_rkey_unpack(uct_md_component_t *mdc,
                                         const void *rkey_buffer, uct_rkey_t *rkey_p,
                                         void **handle_p)
{
    *rkey_p   = 0xdeadbeef;
    *handle_p = NULL;
    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_rkey_release(uct_md_component_t *mdc, uct_rkey_t rkey,
                                          void *handle)
{
    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_mem_reg(uct_md_h md, void *address, size_t length,
                                     unsigned flags, uct_mem_h *memh_p)
{
    cudaError_t cuerr = cudaSuccess;

    if(address == NULL) {
        *memh_p = address;
        return UCS_OK;
    }

    cuerr = cudaHostRegister(address, length, cudaHostRegisterPortable);
    if (cuerr != cudaSuccess) {
        return UCS_ERR_IO_ERROR;
    }

    *memh_p = address;
    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    void *address = (void *)memh;
    cudaError_t cuerr = cudaSuccess;
    if (address == NULL) {
        return UCS_OK;
    }
    cuerr = cudaHostUnregister(address);
    if (cuerr != cudaSuccess) {
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                unsigned *num_resources_p)
{
    return uct_single_md_resource(&uct_cuda_copy_md, resources_p, num_resources_p);
}

static ucs_status_t uct_cuda_copy_md_open(const char *md_name, const uct_md_config_t *md_config,
                                     uct_md_h *md_p)
{
    static uct_md_ops_t md_ops = {
        .close        = (void*)ucs_empty_function,
        .query        = uct_cuda_copy_md_query,
        .mkey_pack    = uct_cuda_copy_mkey_pack,
        .mem_reg      = uct_cuda_copy_mem_reg,
        .mem_dereg    = uct_cuda_copy_mem_dereg
    };
    static uct_md_t md = {
        .ops          = &md_ops,
        .component    = &uct_cuda_copy_md
    };

    *md_p = &md;
    return UCS_OK;
}

UCT_MD_COMPONENT_DEFINE(uct_cuda_copy_md, UCT_CUDA_MD_NAME,
                        uct_cuda_copy_query_md_resources, uct_cuda_copy_md_open, NULL,
                        uct_cuda_copy_rkey_unpack, uct_cuda_copy_rkey_release, "CUDA_",
                        uct_md_config_table, uct_md_config_t);


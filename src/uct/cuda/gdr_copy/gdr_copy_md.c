/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "gdr_copy_md.h"

#include <string.h>
#include <limits.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>
#include <cuda_runtime.h>
#include <cuda.h>

static ucs_config_field_t uct_gdr_copy_md_config_table[] = {
    {"", "", NULL,
    ucs_offsetof(uct_gdr_copy_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {NULL}
};

static ucs_status_t uct_gdr_copy_md_query(uct_md_h md, uct_md_attr_t *md_attr)
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

static ucs_status_t uct_gdr_copy_mkey_pack(uct_md_h md, uct_mem_h memh,
                                      void *rkey_buffer)
{
    return UCS_OK;
}

static ucs_status_t uct_gdr_copy_rkey_unpack(uct_md_component_t *mdc,
                                         const void *rkey_buffer, uct_rkey_t *rkey_p,
                                         void **handle_p)
{
    *rkey_p   = 0xdeadbeef;
    *handle_p = NULL;
    return UCS_OK;
}

static ucs_status_t uct_gdr_copy_rkey_release(uct_md_component_t *mdc, uct_rkey_t rkey,
                                          void *handle)
{
    return UCS_OK;
}


static ucs_status_t uct_gdr_copy_mem_reg(uct_md_h uct_md, void *address, size_t length,
                                     unsigned flags, uct_mem_h *memh_p)
{
    uct_gdr_copy_mem_h * mem_hndl = NULL;
    uct_gdr_copy_md_t *md = ucs_derived_of(uct_md, uct_gdr_copy_md_t);
    gdr_mh_t mh;
    size_t reg_size;
    void *bar_ptr;
    
    CUdeviceptr d_ptr = ((CUdeviceptr )(char *) address);

    mem_hndl = ucs_malloc(sizeof(uct_gdr_copy_mem_h), "gdr_copy handle");
    if (NULL == mem_hndl) {
      ucs_error("Failed to allocate memory for uct_gdr_copy_mem_h");
      return UCS_ERR_NO_MEMORY;
    }

    reg_size = (length + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

    if (gdr_pin_buffer(md->gdrcpy_ctx, (d_ptr & GPU_PAGE_MASK), reg_size, 0, 0, &mh) != 0) {
        ucs_error("gdr_pin_buffer Failed. length :%lu pin_size:%lu ", length, reg_size);
        return UCS_ERR_IO_ERROR;
        
    }
    if (mh == 0) {
        ucs_error("gdr_pin_buffer Failed. length :%lu pin_size:%lu ", length, reg_size);
        return UCS_ERR_IO_ERROR;
    }

    if (gdr_map(md->gdrcpy_ctx, mh, &bar_ptr, reg_size) !=0) {
        ucs_error("gdr_map failed. length :%lu pin_size:%lu ", length, reg_size);
        return UCS_ERR_IO_ERROR;
    }

    mem_hndl->mh = mh;
    mem_hndl->bar_ptr = bar_ptr;
    mem_hndl->reg_size = reg_size;

    *memh_p = mem_hndl;
    return UCS_OK;
}

static ucs_status_t uct_gdr_copy_mem_dereg(uct_md_h uct_md, uct_mem_h memh)
{
    uct_gdr_copy_md_t *md = ucs_derived_of(uct_md, uct_gdr_copy_md_t);
    uct_gdr_copy_mem_h *mem_hndl = memh;

    if (gdr_unmap(md->gdrcpy_ctx, mem_hndl->mh, mem_hndl->bar_ptr, mem_hndl->reg_size) !=0) {
        ucs_error("gdr_unmap Failed. unpin_size:%lu ", mem_hndl->reg_size);
        return UCS_ERR_IO_ERROR;
    }
    if (gdr_unpin_buffer(md->gdrcpy_ctx, mem_hndl->mh) !=0) {
        ucs_error("gdr_unpin_buffer failed ");
        return UCS_ERR_IO_ERROR;
    }

    free(mem_hndl);
    return UCS_OK;
}

static ucs_status_t uct_gdr_copy_mem_detect(uct_md_h md, void *addr, uint64_t *dn_mask)
{
    int memory_type;
    cudaError_t cuda_err = cudaSuccess;
    struct cudaPointerAttributes attributes;
    CUresult cu_err = CUDA_SUCCESS;

    (*dn_mask) = 0;

    if (addr == NULL) {
        return UCS_OK;
    }

    cu_err = cuPointerGetAttribute(&memory_type,
                                   CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                   (CUdeviceptr)addr);
    if (cu_err != CUDA_SUCCESS) {
        cuda_err = cudaPointerGetAttributes (&attributes, addr);
        if (cuda_err == cudaSuccess) {
            if (attributes.memoryType == cudaMemoryTypeDevice) {
                (*dn_mask) = UCT_MD_ADDR_DOMAIN_CUDA;
            }
        }
    } else if (memory_type == CU_MEMORYTYPE_DEVICE) {
        (*dn_mask) = UCT_MD_ADDR_DOMAIN_CUDA;
    }

    return UCS_OK;
}

static ucs_status_t uct_gdr_copy_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                unsigned *num_resources_p)
{

    return uct_single_md_resource(&uct_gdr_copy_md_component, resources_p, num_resources_p);
}

static void uct_gdr_copy_md_close(uct_md_h uct_md)
{
    uct_gdr_copy_md_t *md = ucs_derived_of(uct_md, uct_gdr_copy_md_t);

    if (gdr_close(md->gdrcpy_ctx) != 0) {
        ucs_error("Failed to close gdrcopy");
    }

    ucs_free(md);
}

static ucs_status_t uct_gdr_copy_md_open(const char *md_name, const uct_md_config_t *md_config,
                                     uct_md_h *md_p)
{
    uct_gdr_copy_md_t *md;

    static uct_md_ops_t md_ops = {
        .close        = uct_gdr_copy_md_close,
        .query        = uct_gdr_copy_md_query,
        .mkey_pack    = uct_gdr_copy_mkey_pack,
        .mem_reg      = uct_gdr_copy_mem_reg,
        .mem_dereg    = uct_gdr_copy_mem_dereg,
        .mem_detect   = uct_gdr_copy_mem_detect
    };

    md = ucs_malloc(sizeof(uct_gdr_copy_md_t), "uct_gdr_copy_md_t");
    if (NULL == md) {
        ucs_error("Failed to allocate memory for uct_gdr_copy_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops = &md_ops;
    md->super.component = &uct_gdr_copy_md_component;

    md->gdrcpy_ctx = gdr_open();
    if (md->gdrcpy_ctx == (void *)0) {
        ucs_error("Failed to open gdrcopy ");
        return UCS_ERR_IO_ERROR;
    }

    *md_p = (uct_md_h) md;
    return UCS_OK;
}

UCT_MD_COMPONENT_DEFINE(uct_gdr_copy_md_component, UCT_GDR_COPY_MD_NAME,
                        uct_gdr_copy_query_md_resources, uct_gdr_copy_md_open, NULL,
                        uct_gdr_copy_rkey_unpack, uct_gdr_copy_rkey_release, "GDR_COPY_MD_",
                        uct_gdr_copy_md_config_table, uct_gdr_copy_md_config_t);


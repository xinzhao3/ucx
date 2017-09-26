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

#define UCT_GDR_COPY_MD_RCACHE_DEFAULT_ALIGN 4096

static ucs_config_field_t uct_gdr_copy_md_config_table[] = {
    {"", "", NULL,
        ucs_offsetof(uct_gdr_copy_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"RCACHE", "try", "Enable using memory registration cache",
        ucs_offsetof(uct_gdr_copy_md_config_t, rcache.enable), UCS_CONFIG_TYPE_TERNARY},

    {"RCACHE_ADDR_ALIGN", UCS_PP_MAKE_STRING(UCT_GDR_COPY_MD_RCACHE_DEFAULT_ALIGN),
        "Registration cache address alignment, must be power of 2\n"
            "between "UCS_PP_MAKE_STRING(UCS_PGT_ADDR_ALIGN)"and system page size",
        ucs_offsetof(uct_gdr_copy_md_config_t, rcache.alignment), UCS_CONFIG_TYPE_UINT},

    {"RCACHE_MEM_PRIO", "1000", "Registration cache memory event priority",
        ucs_offsetof(uct_gdr_copy_md_config_t, rcache.event_prio), UCS_CONFIG_TYPE_UINT},

    {"RCACHE_OVERHEAD", "90ns", "Registration cache lookup overhead",
        ucs_offsetof(uct_gdr_copy_md_config_t, rcache.overhead), UCS_CONFIG_TYPE_TIME},

    {"MEM_REG_OVERHEAD", "16us", "Memory registration overhead", /* TODO take default from device */
        ucs_offsetof(uct_gdr_copy_md_config_t, uc_reg_cost.overhead), UCS_CONFIG_TYPE_TIME},

    {"MEM_REG_GROWTH", "0.06ns", "Memory registration growth rate", /* TODO take default from device */
        ucs_offsetof(uct_gdr_copy_md_config_t, uc_reg_cost.growth), UCS_CONFIG_TYPE_TIME},

    {NULL}
};

static ucs_status_t uct_gdr_copy_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags         = UCT_MD_FLAG_REG | UCT_MD_FLAG_ADDR_DN;
    md_attr->cap.addr_dn       = UCT_MD_ADDR_DOMAIN_CUDA;
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

static ucs_status_t uct_gdr_copy_mem_reg_internal(uct_md_h uct_md, void *address, size_t length,
                                     unsigned flags, uct_gdr_copy_mem_t *mem_hndl)
{
    uct_gdr_copy_md_t *md = ucs_derived_of(uct_md, uct_gdr_copy_md_t);
    CUdeviceptr d_ptr = ((CUdeviceptr )(char *) address);
    gdr_mh_t mh;
    void *bar_ptr;

    if (gdr_pin_buffer(md->gdrcpy_ctx, d_ptr, length, 0, 0, &mh) != 0) {
        ucs_error("gdr_pin_buffer Failed. length :%lu ", length);
        return UCS_ERR_IO_ERROR;

    }
    if (mh == 0) {
        ucs_error("gdr_pin_buffer Failed. length :%lu ", length);
        return UCS_ERR_IO_ERROR;
    }

    if (gdr_map(md->gdrcpy_ctx, mh, &bar_ptr, length) !=0) {
        ucs_error("gdr_map failed. length :%lu ", length);
        return UCS_ERR_IO_ERROR;
    }

    mem_hndl->mh = mh;
    mem_hndl->bar_ptr = bar_ptr;
    mem_hndl->reg_size = length;

    return UCS_OK;

}

static ucs_status_t uct_gdr_copy_mem_dereg_internal(uct_md_h uct_md, uct_gdr_copy_mem_t *mem_hndl)
{

    uct_gdr_copy_md_t *md = ucs_derived_of(uct_md, uct_gdr_copy_md_t);

    if (gdr_unmap(md->gdrcpy_ctx, mem_hndl->mh, mem_hndl->bar_ptr, mem_hndl->reg_size) !=0) {
        ucs_error("gdr_unmap Failed. unpin_size:%lu ", mem_hndl->reg_size);
        return UCS_ERR_IO_ERROR;
    }
    if (gdr_unpin_buffer(md->gdrcpy_ctx, mem_hndl->mh) !=0) {
        ucs_error("gdr_unpin_buffer failed ");
        return UCS_ERR_IO_ERROR;
    }
    return UCS_OK;
}

static ucs_status_t uct_gdr_copy_mem_reg(uct_md_h uct_md, void *address, size_t length,
                                     unsigned flags, uct_mem_h *memh_p)
{
    uct_gdr_copy_mem_t * mem_hndl = NULL;
    size_t reg_size;
    void *ptr;
    ucs_status_t status;


    mem_hndl = ucs_malloc(sizeof(uct_gdr_copy_mem_t), "gdr_copy handle");
    if (NULL == mem_hndl) {
      ucs_error("Failed to allocate memory for uct_gdr_copy_mem_t");
      return UCS_ERR_NO_MEMORY;
    }

    reg_size = (length + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    ptr = (void *) ((uintptr_t)address & GPU_PAGE_MASK);

    status = uct_gdr_copy_mem_reg_internal(uct_md, ptr, reg_size, 0, mem_hndl);
    if (status != UCS_OK) {
        free(mem_hndl);
        return status;
    }

    *memh_p = mem_hndl;
    return UCS_OK;
}

static ucs_status_t uct_gdr_copy_mem_dereg(uct_md_h uct_md, uct_mem_h memh)
{
    uct_gdr_copy_mem_t *mem_hndl = memh;
    ucs_status_t status;

    status = uct_gdr_copy_mem_dereg_internal(uct_md, mem_hndl);
    free(mem_hndl);
    return status;
}

static ucs_status_t uct_gdr_copy_mem_detect(uct_md_h md, void *addr)
{
    int memory_type;
    cudaError_t cuda_err = cudaSuccess;
    struct cudaPointerAttributes attributes;
    CUresult cu_err = CUDA_SUCCESS;

    if (addr == NULL) {
        return UCS_ERR_INVALID_ADDR;
    }

    cu_err = cuPointerGetAttribute(&memory_type,
                                   CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                   (CUdeviceptr)addr);
    if (cu_err != CUDA_SUCCESS) {
        cuda_err = cudaPointerGetAttributes (&attributes, addr);
        if (cuda_err == cudaSuccess) {
            if (attributes.memoryType == cudaMemoryTypeDevice) {
                return UCS_OK;
            }
        }
    } else if (memory_type == CU_MEMORYTYPE_DEVICE) {
        return UCS_OK;
    }

    return UCS_ERR_INVALID_ADDR;
}

static ucs_status_t uct_gdr_copy_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                unsigned *num_resources_p)
{

    return uct_single_md_resource(&uct_gdr_copy_md_component, resources_p, num_resources_p);
}

static void uct_gdr_copy_md_close(uct_md_h uct_md)
{
    uct_gdr_copy_md_t *md = ucs_derived_of(uct_md, uct_gdr_copy_md_t);

    if (md->rcache != NULL) {
        ucs_rcache_destroy(md->rcache);
    }

    if (gdr_close(md->gdrcpy_ctx) != 0) {
        ucs_error("Failed to close gdrcopy");
    }

    ucs_free(md);
}

static uct_md_ops_t md_ops = {
    .close        = uct_gdr_copy_md_close,
    .query        = uct_gdr_copy_md_query,
    .mkey_pack    = uct_gdr_copy_mkey_pack,
    .mem_reg      = uct_gdr_copy_mem_reg,
    .mem_dereg    = uct_gdr_copy_mem_dereg,
    .mem_detect   = uct_gdr_copy_mem_detect
};

static inline uct_gdr_copy_rcache_region_t* uct_gdr_copy_rache_region_from_memh(uct_mem_h memh)
{
    return ucs_container_of(memh, uct_gdr_copy_rcache_region_t, memh);
}

static ucs_status_t uct_gdr_copy_mem_rcache_reg(uct_md_h uct_md, void *address,
                                          size_t length, unsigned flags,
                                          uct_mem_h *memh_p)
{
    uct_gdr_copy_md_t *md = ucs_derived_of(uct_md, uct_gdr_copy_md_t);
    ucs_rcache_region_t *rregion;
    ucs_status_t status;
    uct_gdr_copy_mem_t *memh;

    status = ucs_rcache_get(md->rcache, address, length, PROT_READ|PROT_WRITE,
                            &flags, &rregion);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assert(rregion->refcount > 0);
    memh = &ucs_derived_of(rregion, uct_gdr_copy_rcache_region_t)->memh;
    *memh_p = memh;
    return UCS_OK;
}

static ucs_status_t uct_gdr_copy_mem_rcache_dereg(uct_md_h uct_md, uct_mem_h memh)
{
    uct_gdr_copy_md_t *md = ucs_derived_of(uct_md, uct_gdr_copy_md_t);
    uct_gdr_copy_rcache_region_t *region = uct_gdr_copy_rache_region_from_memh(memh);

    ucs_rcache_region_put(md->rcache, &region->super);
    return UCS_OK;
}

static uct_md_ops_t md_rcache_ops = {
    .close        = uct_gdr_copy_md_close,
    .query        = uct_gdr_copy_md_query,
    .mkey_pack    = uct_gdr_copy_mkey_pack,
    .mem_reg      = uct_gdr_copy_mem_rcache_reg,
    .mem_dereg    = uct_gdr_copy_mem_rcache_dereg,
    .mem_detect   = uct_gdr_copy_mem_detect
};
static ucs_status_t uct_gdr_copy_rcache_mem_reg_cb(void *context, ucs_rcache_t *rcache,
                                             void *arg, ucs_rcache_region_t *rregion)
{
    uct_gdr_copy_rcache_region_t *region = ucs_derived_of(rregion, uct_gdr_copy_rcache_region_t);
    uct_gdr_copy_md_t *md = context;
    int *flags = arg;
    ucs_status_t status;

    status = uct_gdr_copy_mem_reg_internal(&md->super, (void*)region->super.super.start,
                                     region->super.super.end - region->super.super.start,
                                     *flags, &region->memh);
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}

static void uct_gdr_copy_rcache_mem_dereg_cb(void *context, ucs_rcache_t *rcache,
                                       ucs_rcache_region_t *rregion)
{
    uct_gdr_copy_rcache_region_t *region = ucs_derived_of(rregion, uct_gdr_copy_rcache_region_t);
    uct_gdr_copy_md_t *md = context;

    (void)uct_gdr_copy_mem_dereg_internal(&md->super, &region->memh);
}

static void uct_gdr_copy_rcache_dump_region_cb(void *context, ucs_rcache_t *rcache,
                                         ucs_rcache_region_t *rregion, char *buf,
                                         size_t max)
{

}

static ucs_rcache_ops_t uct_gdr_copy_rcache_ops = {
    .mem_reg     = uct_gdr_copy_rcache_mem_reg_cb,
    .mem_dereg   = uct_gdr_copy_rcache_mem_dereg_cb,
    .dump_region = uct_gdr_copy_rcache_dump_region_cb
};

static ucs_status_t uct_gdr_copy_md_open(const char *md_name, const uct_md_config_t *uct_md_config,
        uct_md_h *md_p)
{
    ucs_status_t status;
    uct_gdr_copy_md_t *md;
    const uct_gdr_copy_md_config_t *md_config = ucs_derived_of(uct_md_config, uct_gdr_copy_md_config_t);
    ucs_rcache_params_t rcache_params;

    md = ucs_malloc(sizeof(uct_gdr_copy_md_t), "uct_gdr_copy_md_t");
    if (NULL == md) {
        ucs_error("Failed to allocate memory for uct_gdr_copy_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops = &md_ops;
    md->super.component = &uct_gdr_copy_md_component;
    md->rcache = NULL;
    md->reg_cost = md_config->uc_reg_cost;



    md->gdrcpy_ctx = gdr_open();
    if (md->gdrcpy_ctx == (void *)0) {
        ucs_error("Failed to open gdrcopy ");
        return UCS_ERR_IO_ERROR;
    }

    if (md_config->rcache.enable != UCS_NO) {
       // UCS_STATIC_ASSERT(UCS_PGT_ADDR_ALIGN >= UCT_GDR_COPY_MD_RCACHE_DEFAULT_ALIGN);
        rcache_params.region_struct_size = sizeof(uct_gdr_copy_rcache_region_t);
        rcache_params.alignment          = md_config->rcache.alignment;
        rcache_params.ucm_event_priority = md_config->rcache.event_prio;
        rcache_params.context            = md;
        rcache_params.ops                = &uct_gdr_copy_rcache_ops;
        status = ucs_rcache_create(&rcache_params, "gdr_copy" UCS_STATS_ARG(NULL), &md->rcache);
        if (status == UCS_OK) {
            md->super.ops         = &md_rcache_ops;
            md->reg_cost.overhead = 0;
            md->reg_cost.growth   = 0; /* It's close enough to 0 */
        } else {
            ucs_assert(md->rcache == NULL);
            if (md_config->rcache.enable == UCS_YES) {
                ucs_error("Failed to create registration cache: %s",
                          ucs_status_string(status));
                return UCS_ERR_IO_ERROR;
            } else {
                ucs_debug("Could not create registration cache for: %s",
                          ucs_status_string(status));
            }
        }
    }

    *md_p = (uct_md_h) md;
    return UCS_OK;
}

UCT_MD_COMPONENT_DEFINE(uct_gdr_copy_md_component, UCT_GDR_COPY_MD_NAME,
                        uct_gdr_copy_query_md_resources, uct_gdr_copy_md_open, NULL,
                        uct_gdr_copy_rkey_unpack, uct_gdr_copy_rkey_release, "GDR_COPY_",
                        uct_gdr_copy_md_config_table, uct_gdr_copy_md_config_t);


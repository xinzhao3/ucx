/**
 * Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "gdr_copy_ep.h"
#include "gdr_copy_md.h"
#include "gdr_copy_iface.h"

#include <uct/base/uct_log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>


static UCS_CLASS_INIT_FUNC(uct_gdr_copy_ep_t, uct_iface_t *tl_iface,
                           const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)
{
    uct_gdr_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_gdr_copy_iface_t);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super)
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_gdr_copy_ep_t)
{
}

UCS_CLASS_DEFINE(uct_gdr_copy_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_gdr_copy_ep_t, uct_ep_t, uct_iface_t*,
                          const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_gdr_copy_ep_t, uct_ep_t);


ucs_status_t uct_gdr_copy_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    uct_gdr_copy_iface_t *iface   = ucs_derived_of(tl_ep->iface, uct_gdr_copy_iface_t);
    uct_gdr_copy_md_t *md = (uct_gdr_copy_md_t *)iface->super.md;
    uct_gdr_copy_mem_t *mem_hndl = (uct_gdr_copy_mem_t *) rkey;
    gdr_info_t gdr_info;
    size_t bar_off;

    assert(iovcnt == 1);

    if (gdr_get_info(md->gdrcpy_ctx, mem_hndl->mh, &gdr_info) != 0) {
        ucs_error("gdr_get_info failed. ");
        return UCS_ERR_IO_ERROR;
    }
    bar_off = remote_addr - gdr_info.va;

    gdr_copy_to_bar ((mem_hndl->bar_ptr + bar_off), iov[0].buffer, iov[0].length);

    return UCS_OK;
}

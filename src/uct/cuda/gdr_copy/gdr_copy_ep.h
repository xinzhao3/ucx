/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCT_SYSV_EP_H
#define UCT_SYSV_EP_H

#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/type/class.h>


typedef struct uct_gdr_copy_ep_addr {
    int                ep_id;
} uct_gdr_copy_ep_addr_t;

typedef struct uct_gdr_copy_ep {
    uct_base_ep_t      super;
    struct uct_gdr_copy_ep *next;
} uct_gdr_copy_ep_t;

UCS_CLASS_DECLARE_NEW_FUNC(uct_gdr_copy_ep_t, uct_ep_t, uct_iface_t*,
                           const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_gdr_copy_ep_t, uct_ep_t);

ucs_status_t uct_gdr_copy_ep_put_zcopy(uct_ep_h tl_ep,
                                       const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp);

#endif

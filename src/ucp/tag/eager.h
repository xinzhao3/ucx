/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TAG_EAGER_H_
#define UCP_TAG_EAGER_H_

#include "tag_match.h"

#include <ucp/api/ucp.h>
#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_request.h>
#include <ucp/dt/dt.inl>
#include <ucp/proto/proto.h>


/*
 * EAGER_ONLY, EAGER_MIDDLE, EAGER_LAST
 */
typedef struct {
    ucp_tag_hdr_t             super;
    /* TODO offset/sequence number */
} UCS_S_PACKED ucp_eager_hdr_t;


/*
 * EAGER_FIRST
 */
typedef struct {
    ucp_eager_hdr_t           super;
    size_t                    total_len;
} UCS_S_PACKED ucp_eager_first_hdr_t;


/*
 * EAGER_SYNC_ONLY
 */
typedef struct {
    ucp_eager_hdr_t           super;
    ucp_request_hdr_t         req;
} UCS_S_PACKED ucp_eager_sync_hdr_t;


/*
 * EAGER_SYNC_FIRST
 */
typedef struct {
    ucp_eager_first_hdr_t     super;
    ucp_request_hdr_t         req;
} UCS_S_PACKED ucp_eager_sync_first_hdr_t;


extern const ucp_proto_t ucp_tag_eager_proto;
extern const ucp_proto_t ucp_tag_eager_sync_proto;


void ucp_tag_eager_sync_send_ack(ucp_worker_h worker, uint64_t sender_uuid,
                                 uintptr_t remote_request);

void ucp_tag_eager_sync_completion(ucp_request_t *req, uint16_t flag,
                                   ucs_status_t status);

void ucp_eager_sync_send_handler(void *arg, void *data, uint16_t flags);

void ucp_tag_eager_zcopy_completion(uct_completion_t *self, ucs_status_t status);

void ucp_tag_eager_zcopy_req_complete(ucp_request_t *req, ucs_status_t status);

void ucp_tag_eager_sync_zcopy_req_complete(ucp_request_t *req, ucs_status_t status);

void ucp_tag_eager_sync_zcopy_completion(uct_completion_t *self, ucs_status_t status);

static inline ucs_status_t ucp_tag_send_eager_short(ucp_ep_t *ep, ucp_tag_t tag,
                                                    const void *buffer, size_t length)
{
    if (ep->flags & UCP_EP_FLAG_TAG_OFFLOAD_ENABLED) {
        UCS_STATIC_ASSERT(sizeof(ucp_tag_t) == sizeof(uct_tag_t));
        return uct_ep_tag_eager_short(ucp_ep_get_tag_uct_ep(ep), tag, buffer, length);
    } else {
        UCS_STATIC_ASSERT(sizeof(ucp_tag_t) == sizeof(ucp_eager_hdr_t));
        UCS_STATIC_ASSERT(sizeof(ucp_tag_t) == sizeof(uint64_t));
        return uct_ep_am_short(ucp_ep_get_am_uct_ep(ep), UCP_AM_ID_EAGER_ONLY, tag,
                               buffer, length);
    }
}

static UCS_F_ALWAYS_INLINE size_t
ucp_eager_total_len(ucp_eager_hdr_t *hdr, unsigned flags, unsigned payload_length)
{
    ucs_assert(flags & UCP_RECV_DESC_FLAG_FIRST);
    if (flags & UCP_RECV_DESC_FLAG_LAST) {
        return payload_length;
    } else {
        return ucs_container_of(hdr, ucp_eager_first_hdr_t, super)->total_len;
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_eager_unexp_match(ucp_worker_h worker, ucp_recv_desc_t *rdesc, ucp_tag_t tag,
                      unsigned flags, void *buffer, size_t count,
                      uct_memory_type_t mem_type, ucp_datatype_t datatype,
                      ucp_dt_state_t *state, ucp_tag_recv_info_t *info)
{
    size_t recv_len, hdr_len;
    ucs_status_t status;
    void *data = rdesc + 1;

    UCP_WORKER_STAT_EAGER_CHUNK(worker, UNEXP);
    hdr_len  = rdesc->payload_offset;
    recv_len = rdesc->length - hdr_len;
    status   = ucp_dt_unpack(worker, datatype, buffer, count, mem_type, state,
                             data + hdr_len, recv_len, flags & UCP_RECV_DESC_FLAG_LAST);
    state->offset += recv_len;

    if (flags & UCP_RECV_DESC_FLAG_FIRST) {
        info->sender_tag = tag;
        info->length     = ucp_eager_total_len(data, flags, recv_len);

        if (ucs_unlikely(flags & UCP_RECV_DESC_FLAG_SYNC)) {
            ucp_eager_sync_send_handler(worker, data, flags);
        }
        UCP_WORKER_STAT_EAGER_MSG(worker, flags);
    }

    if (flags & UCP_RECV_DESC_FLAG_LAST) {
        info->length     = state->offset;
        return status;
    }

    return UCS_INPROGRESS;
}

#endif

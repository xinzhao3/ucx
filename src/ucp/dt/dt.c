/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "dt.h"

#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_request.h>
#include <ucs/debug/profile.h>

size_t ucp_dt_pack(ucp_datatype_t datatype, void *dest, const void *src,
                   ucp_dt_state_t *state, size_t length)
{
    ucp_dt_generic_t *dt;
    size_t result_len = 0;

    if (!length) {
        return length;
    }

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        UCS_PROFILE_CALL(memcpy, dest, src + state->offset, length);
        result_len = length;
        break;

    case UCP_DATATYPE_IOV:
        UCS_PROFILE_CALL_VOID(ucp_dt_iov_gather, dest, src, length,
                              &state->dt.iov.iov_offset,
                              &state->dt.iov.iovcnt_offset);
        result_len = length;
        break;

    case UCP_DATATYPE_GENERIC:
        dt = ucp_dt_generic(datatype);
        result_len = UCS_PROFILE_NAMED_CALL("dt_pack", dt->ops.pack,
                                            state->dt.generic.state,
                                            state->offset, dest, length);
        break;

    default:
        ucs_error("Invalid data type");
    }

    state->offset += result_len;
    return result_len;
}

#define MAX_RKEY_BUFFER (256)

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_mem_type_unpack(ucp_worker_h worker, void *buffer, const void *recv_data,
                    size_t recv_length, uct_memory_type_t mem_type)
{
    ucp_context_h context = worker->context;
    ucp_ep_h ep = worker->mem_type_ep[mem_type];
    ucp_md_map_t md_map = 0;
    ucp_lane_index_t lane;
    uct_md_h md;
    unsigned md_index;
    uct_mem_h memh[1];
    ucs_status_t status;
    char rkey_buffer[MAX_RKEY_BUFFER];
    uct_rkey_bundle_t rkey_bundle;

    if (recv_length == 0) {
        return UCS_OK;
    }

    lane = ucp_ep_config(ep)->key.rma_lanes[0];
    md_index = ucp_ep_md_index(ep, lane);
    md = context->tl_mds[md_index].md;
    ucs_assert(MAX_RKEY_BUFFER > context->tl_mds[md_index].attr.rkey_packed_size);

    status = ucp_mem_rereg_mds(context, UCS_BIT(md_index), buffer,
                               recv_length, UCT_MD_MEM_ACCESS_ALL, NULL, mem_type,
                               NULL, memh, &md_map);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_md_mkey_pack(md, memh[0], rkey_buffer);
    if (status != UCS_OK) {
        ucs_error("failed to pack key from md[%d]: %s",
                  md_index, ucs_status_string(status));
        goto err_dreg_mem;
    }

    status = uct_rkey_unpack(rkey_buffer, &rkey_bundle);
    if (status != UCS_OK) {
        ucs_error("failed to unpack key from md[%d]: %s",
                   md_index, ucs_status_string(status));
        goto err_dreg_mem;
    }

    status = uct_ep_put_short(ep->uct_eps[lane], recv_data, recv_length,
                              (uint64_t)buffer, rkey_bundle.rkey);
    if (status != UCS_OK) {
        ucs_error("uct_ep_put_short() failed %s", ucs_status_string(status));
        goto err_dreg_mem;
    }

err_dreg_mem:
    ucp_mem_rereg_mds(context, 0, NULL, 0, 0, NULL, mem_type, NULL, memh, &md_map);
err:
    return status;
}

UCS_F_ALWAYS_INLINE ucs_status_t
ucp_dt_unpack(ucp_worker_h worker, ucp_datatype_t datatype, void *buffer, size_t buffer_size,
              uct_memory_type_t mem_type, ucp_dt_state_t *state, const void *recv_data,
              size_t recv_length, unsigned flags)
{
    ucp_dt_generic_t *dt_gen;
    size_t           offset = state->offset;
    ucs_status_t     status = UCS_OK;

    if (ucs_unlikely((recv_length + offset) > buffer_size)) {
        ucs_debug("message truncated: recv_length %zu offset %zu buffer_size %zu",
                  recv_length, offset, buffer_size);
        if (UCP_DT_IS_GENERIC(datatype) && (flags & UCP_RECV_DESC_FLAG_LAST)) {
            ucp_dt_generic(datatype)->ops.finish(state->dt.generic.state);
        }

        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        if (ucs_likely(UCP_MEM_IS_HOST(mem_type))) {
            UCS_PROFILE_NAMED_CALL("memcpy_recv", memcpy, buffer + offset,
                                   recv_data, recv_length);
        } else {
            status = ucp_mem_type_unpack(worker, buffer + offset, recv_data,
                                         recv_length, mem_type);
        }
        return status;

    case UCP_DATATYPE_IOV:
        UCS_PROFILE_CALL(ucp_dt_iov_scatter, buffer, state->dt.iov.iovcnt,
                         recv_data, recv_length, &state->dt.iov.iov_offset,
                         &state->dt.iov.iovcnt_offset);
        return status;

    case UCP_DATATYPE_GENERIC:
        dt_gen = ucp_dt_generic(datatype);
        status = UCS_PROFILE_NAMED_CALL("dt_unpack", dt_gen->ops.unpack,
                                        state->dt.generic.state, offset,
                                        recv_data, recv_length);
        if (flags & UCP_RECV_DESC_FLAG_LAST) {
            UCS_PROFILE_NAMED_CALL_VOID("dt_finish", dt_gen->ops.finish,
                                        state->dt.generic.state);
        }
        return status;

    default:
        ucs_error("unexpected datatype=%lx", datatype);
        return UCS_ERR_INVALID_PARAM;
    }
}

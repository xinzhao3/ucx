/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "dt.h"
#include <ucp/core/ucp_request.inl>


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

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_dn_dt_unpack(ucp_request_t *req, void *buffer, size_t buffer_size,
        const void *recv_data, size_t recv_length)
{
    ucs_status_t status;
    ucp_worker_h worker = req->recv.worker;
    ucp_context_h context = worker->context;
    ucp_ep_h ep = ucp_worker_ep_find(worker, worker->uuid);
    ucp_ep_config_t *config = ucp_ep_config(ep);
    ucp_md_map_t dn_md_map = req->addr_dn_h->md_map;
    ucp_lane_index_t dn_lane;
    ucp_rsc_index_t rsc_index;
    uct_iface_attr_t *iface_attr;
    unsigned md_index;
    uct_mem_h memh;
    uct_iov_t iov;

    if (recv_length == 0) {
        return UCS_OK;
    }

    while(1) {
        dn_lane = ucp_config_find_domain_lane(config, config->key.domain_lanes, dn_md_map);
        if (dn_lane == UCP_NULL_LANE) {
            ucs_error("Not find address domain lane.");
            return UCS_ERR_IO_ERROR;
        }
        rsc_index    = ucp_ep_get_rsc_index(ep, dn_lane);
        iface_attr = &worker->ifaces[rsc_index].attr;
        md_index = config->key.lanes[dn_lane].dst_md_index;
        if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_PUT_ZCOPY)) {
            dn_md_map |= ~UCS_BIT(md_index);
            continue;
        }
        break;
    }


    status = uct_md_mem_reg(context->tl_mds[md_index].md, buffer, buffer_size,
                        UCT_MD_MEM_ACCESS_REMOTE_PUT, &memh);
    if (status != UCS_OK) {
        ucs_error("Failed to reg address %p with md %s", buffer,
                context->tl_mds[md_index].rsc.md_name);
        return status;
    }

    ucs_assert(buffer_size >= recv_length);
    iov.buffer = (void *)recv_data;
    iov.length = recv_length;
    iov.count  = 1;
    iov.memh   = UCT_MEM_HANDLE_NULL;


    status = uct_ep_put_zcopy(ep->uct_eps[dn_lane], &iov, 1, (uint64_t)buffer,
            (uct_rkey_t )memh, NULL);
    if (status != UCS_OK) {
        uct_md_mem_dereg(context->tl_mds[md_index].md, memh);
        ucs_error("Failed to perform uct_ep_put_zcopy to address %p", recv_data);
        return status;
    }

    status = uct_md_mem_dereg(context->tl_mds[md_index].md, memh);
    if (status != UCS_OK) {
        ucs_error("Failed to dereg address %p with md %s", buffer,
                context->tl_mds[md_index].rsc.md_name);
        return status;
    }

    return UCS_OK;
}


ucs_status_t ucp_dt_unpack(ucp_request_t *req, ucp_datatype_t datatype, void *buffer, size_t buffer_size,
                           ucp_dt_state_t *state, const void *recv_data, size_t recv_length, int last)
{
    ucp_dt_generic_t *dt_gen;
    size_t offset = state->offset;
    ucs_status_t status;

    if (ucs_unlikely((recv_length + offset) > buffer_size)) {
        ucs_trace_req("message truncated: recv_length %zu offset %zu buffer_size %zu",
                      recv_length, offset, buffer_size);
        if (UCP_DT_IS_GENERIC(datatype) && last) {
            ucp_dt_generic(datatype)->ops.finish(state->dt.generic.state);
        }
        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        if (ucs_likely(UCP_IS_DEFAULT_ADDR_DOMAIN(req->addr_dn_h))) {
            UCS_PROFILE_NAMED_CALL("memcpy_recv", memcpy, buffer + offset,
                                   recv_data, recv_length);
            return UCS_OK;
        } else {
            return ucp_dn_dt_unpack(req, buffer, buffer_size, recv_data, recv_length);
        }

    case UCP_DATATYPE_IOV:
        UCS_PROFILE_CALL(ucp_dt_iov_scatter, buffer, state->dt.iov.iovcnt,
                         recv_data, recv_length, &state->dt.iov.iov_offset,
                         &state->dt.iov.iovcnt_offset);
        return UCS_OK;

    case UCP_DATATYPE_GENERIC:
        dt_gen = ucp_dt_generic(datatype);
        status = UCS_PROFILE_NAMED_CALL("dt_unpack", dt_gen->ops.unpack,
                                        state->dt.generic.state, offset,
                                        recv_data, recv_length);
        if (last) {
            UCS_PROFILE_NAMED_CALL_VOID("dt_finish", dt_gen->ops.finish,
                                        state->dt.generic.state);
        }
        return status;

    default:
        ucs_error("unexpected datatype=%lx", datatype);
        return UCS_ERR_INVALID_PARAM;
    }
}

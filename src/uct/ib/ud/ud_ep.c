/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 *
 * $COPYRIGHT$
 * $HEADER$
 */

#include "ud_ep.h"
#include "ud_iface.h"

#include <uct/ib/base/ib_verbs.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/type/class.h>

static UCS_CLASS_INIT_FUNC(uct_ud_ep_t, uct_iface_t *tl_iface)
{
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);

    ucs_trace_func("");

    UCS_CLASS_CALL_SUPER_INIT(tl_iface);

    self->dest_ep_id = UCT_UD_EP_NULL_ID;
    self->ah         = NULL;
    uct_ud_iface_add_ep(iface, self);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ud_ep_t)
{
    uct_ud_iface_t *iface = ucs_derived_of(self->super.iface, uct_ud_iface_t);

    ucs_trace_func("");

    if (self->ah) { 
        ibv_destroy_ah(self->ah);
    }
    uct_ud_iface_remove_ep(iface, self);
   /* TODO: in disconnect ucs_frag_list_cleanup(&self->rx.ooo_pkts); */
}

UCS_CLASS_DEFINE(uct_ud_ep_t, uct_ep_t);


ucs_status_t uct_ud_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *ep_addr)
{
    uct_ud_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_ep_t);

    ((uct_ud_ep_addr_t*)ep_addr)->ep_id = ep->ep_id;
    ucs_debug("ep_addr=%d", ep->ep_id);
    return UCS_OK;
}

ucs_status_t uct_ud_ep_connect_to_ep(uct_ep_h tl_ep,
                                     uct_iface_addr_t *tl_iface_addr,
                                     uct_ep_addr_t *tl_ep_addr)
{
    uct_ud_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_ep_t);
    uct_ud_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ud_iface_t);
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super);
    uct_ud_iface_addr_t *if_addr = ucs_derived_of(tl_iface_addr, uct_ud_iface_addr_t);
    uct_ud_ep_addr_t *ep_addr = ucs_derived_of(tl_ep_addr, uct_ud_ep_addr_t);

    struct ibv_ah_attr ah_attr;
    struct ibv_ah *ah;

    ucs_assert_always(ep->dest_ep_id == UCT_UD_EP_NULL_ID);
    ucs_trace_func("");

    memset(&ah_attr, 0, sizeof(ah_attr));
    ah_attr.port_num = iface->super.port_num; 
    ah_attr.sl = 0; /* TODO: sl */
    ah_attr.is_global = 0;
    ah_attr.dlid = if_addr->lid;

    ah = ibv_create_ah(dev->pd, &ah_attr);
    if (ah == NULL) {
        ucs_error("failed to create address handle: %m");
        return UCS_ERR_INVALID_ADDR;
    }

    ep->ah = ah;
    ep->dest_ep_id = ep_addr->ep_id;
    ep->dest_qpn = if_addr->qp_num;

    ep->tx.psn       = 1;
    /* TODO: configurable max window size */
    ep->tx.max_psn   = ep->tx.psn + UCT_UD_MAX_WINDOW;
    ep->tx.acked_psn = 0;
    ucs_callbackq_init(&ep->tx.window);

    ep->rx.acked_psn = 0;
    ucs_frag_list_init(ep->tx.psn-1, &ep->rx.ooo_pkts, 0 /*TODO: ooo support */
                       UCS_STATS_ARG(ep->rx.stats));

    ucs_debug("%s:%d slid=%d qpn=%d ep=%u connected to dlid=%d qpn=%d ep=%u ah=%p", 
              ibv_get_device_name(dev->ibv_context->device),
              iface->super.port_num,
              dev->port_attr[iface->super.port_num-dev->first_port].lid,
              iface->qp->qp_num,
              ep->ep_id, 
              if_addr->lid, if_addr->qp_num, ep->dest_ep_id, ah);


    return UCS_OK;
}

static inline void uct_ud_ep_process_ack(uct_ud_ep_t *ep, uct_ud_psn_t ack_psn)
{

    if (ucs_unlikely(UCT_UD_PSN_COMPARE(ack_psn, <=, ep->tx.acked_psn))) {
        return;
    }

    ep->tx.acked_psn = ack_psn;
    
    /* Release acknowledged skb's */
    ucs_callbackq_pull(&ep->tx.window, ack_psn);

    /* update window */
    ep->tx.max_psn =  ep->tx.acked_psn + UCT_UD_MAX_WINDOW;
}

void uct_ud_ep_process_rx(uct_ud_iface_t *iface, uct_ud_neth_t *neth, unsigned byte_len, uct_ud_recv_skb_t *skb)
{
    uint32_t dest_id;
    uint8_t is_am, am_id;
    uct_ud_ep_t *ep = 0; /* todo: check why gcc complaints about uninitialized var */
    ucs_frag_list_ooo_type_t ooo_type;
    ucs_status_t ret;

    dest_id = uct_ud_neth_get_dest_id(neth);
    am_id   = uct_ud_neth_get_am_id(neth);
    is_am   = neth->packet_type & UCT_UD_PACKET_FLAG_AM;

    ucs_trace_data("src_ep= dest_ep=%d psn=%d ack_psn=%d am_id=%d is_am=%d len=%d packet_type=%08x",
                   dest_id, (int)neth->psn, (int)neth->ack_psn, (int)am_id, (int)is_am, byte_len, neth->packet_type);
    if (ucs_unlikely(!ucs_ptr_array_lookup(&iface->eps, dest_id, ep) ||
                     ep->ep_id != dest_id)) {
        /* TODO: in the future just drop the packet */
        ucs_fatal("Faied to find ep(%d)", dest_id);
        return;
    } 
    ucs_assert(ep->ep_id != UCT_UD_EP_NULL_ID);
    
    /* todo: process ack */
    uct_ud_ep_process_ack(ep, neth->ack_psn);

    if (ucs_unlikely(!is_am)) {
        ucs_fatal("Control packet received - not implemented!");
        return;
    }

    ooo_type = ucs_frag_list_insert(&ep->rx.ooo_pkts, &skb->ooo_elem, neth->psn);
    if (ucs_unlikely(ooo_type != UCS_FRAG_LIST_INSERT_FAST)) {
        ucs_warn("src_ep= dest_ep=%u rx_psn=%hu psn=%hu ack_psn=%hu am_id=%d is_am=%d len=%d",
                 dest_id, ep->rx.ooo_pkts.head_sn, neth->psn, neth->ack_psn, (int)am_id, (int)is_am, byte_len);
        ucs_fatal("Out of order is not implemented: got %d", ooo_type);
        return;
    }

    if (ucs_unlikely(neth->packet_type & UCT_UD_PACKET_FLAG_ACK_REQ)) {
        ucs_fatal("ACK REQ handling is not implemented");
        return;
    }

    ret = uct_iface_invoke_am(&iface->super.super, am_id, skb, neth + 1,
                              byte_len - sizeof(*neth));
    if (ret == UCS_OK) {
        ucs_mpool_put(skb);
    }
}
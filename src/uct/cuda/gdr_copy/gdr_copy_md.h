/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GDR_COPY_MD_H
#define UCT_GDR_COPY_MD_H

#include <uct/base/uct_md.h>
#include <ucs/sys/rcache.h>
#include "gdrapi.h"

#define UCT_GDR_COPY_MD_NAME "gdr_copy"

extern uct_md_component_t uct_gdr_copy_md_component;

/**
 * @brief gdr_copy MD descriptor
 */
typedef struct uct_gdr_copy_md {
    struct uct_md super;   /**< Domain info */
    gdr_t gdrcpy_ctx;      /**< gdr copy context */
    ucs_rcache_t *rcache;  /**< Registration cache (can be NULL) */
    uct_linear_growth_t      reg_cost;  /**< Memory registration cost */
} uct_gdr_copy_md_t;

/**
 * gdr copy domain configuration.
 */
typedef struct uct_gdr_copy_md_config {
    uct_md_config_t super;
    struct {
        ucs_ternary_value_t  enable;       /**< Enable registration cache */
        size_t               alignment;    /**< Force address alignment */
        unsigned             event_prio;   /**< Memory events priority */
        double               overhead;     /**< Lookup overhead estimation */
    } rcache;

    uct_linear_growth_t      uc_reg_cost;  /**< Memory registration cost estimation
                                             without using the cache */


} uct_gdr_copy_md_config_t;


/**
 * @brief gdr copy mem handle
 */
typedef struct uct_gdr_copy_mem {
    gdr_mh_t mh;
    void *bar_ptr;
    size_t reg_size;
} uct_gdr_copy_mem_t;

/**
 * cuda memory region in the registration cache.
 */
typedef struct uct_gdr_copy_rcache_region {
    ucs_rcache_region_t  super;
    uct_gdr_copy_mem_t    memh;      /**<  mr exposed to the user as the memh */
} uct_gdr_copy_rcache_region_t;


#endif
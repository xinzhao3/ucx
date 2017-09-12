/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_CONTEXT_H
#define UCT_CUDA_CONTEXT_H

#include <uct/base/uct_md.h>
#include "gdrapi.h"

#define UCT_GDR_COPY_MD_NAME "gdr_copy"

extern uct_md_component_t uct_gdr_copy_md_component;

/**
 * @brief gdr_copy MD descriptor
 */
typedef struct uct_gdr_copy_md {
    struct uct_md super;   /**< Domain info */
    gdr_t gdrcpy_ctx;  /**< gdr copy context */
} uct_gdr_copy_md_t;

/**
 * gdr copy domain configuration.
 */
typedef struct uct_gdr_copy_md_config {
    uct_md_config_t super;
} uct_gdr_copy_md_config_t;


/**
 * @brief gdr copy mem handle
 */
typedef struct uct_gdr_copy_mem {
    gdr_mh_t mh;
    void *bar_ptr;
    size_t reg_size;
} uct_gdr_copy_mem_h;



#endif

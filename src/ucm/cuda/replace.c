/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cudamem.h"

#include <ucm/event/event.h>
#include <ucm/util/log.h>
#include <ucm/util/reloc.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/type/component.h>
#include <pthread.h>


#define MAP_FAILED ((void*)-1)

static pthread_mutex_t ucm_cudamem_get_orig_lock = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
static pthread_t volatile ucm_cudamem_get_orig_thread = -1;


/**
 * Define a replacement function to a memory-mapping function call, which calls
 * the event handler, and if event handler returns error code - calls the original
 * function.
 */
#define UCM_DEFINE_CUDA_FUNC(_name, _rettype, _fail_val, ...) \
    \
    _rettype ucm_override_##_name(UCM_FUNC_DEFINE_ARGS(__VA_ARGS__)); \
    \
    /* Call the original function using dlsym(RTLD_NEXT) */ \
    _rettype ucm_orig_##_name(UCM_FUNC_DEFINE_ARGS(__VA_ARGS__)) \
    { \
        typedef _rettype (*func_ptr_t) (__VA_ARGS__); \
        static func_ptr_t orig_func_ptr = NULL; \
        \
        ucm_trace("%s()", __FUNCTION__); \
        \
        if (ucs_unlikely(orig_func_ptr == NULL)) { \
            pthread_mutex_lock(&ucm_cudamem_get_orig_lock); \
            ucm_cudamem_get_orig_thread = pthread_self(); \
            orig_func_ptr = ucm_reloc_get_orig(UCS_PP_QUOTE(_name), \
                                               ucm_override_##_name); \
            ucm_cudamem_get_orig_thread = -1; \
            pthread_mutex_unlock(&ucm_cudamem_get_orig_lock); \
        } \
        return orig_func_ptr(UCM_FUNC_PASS_ARGS(__VA_ARGS__)); \
    } \
    \
    /* Define a symbol which goes to the replacement - in case we are loaded first */ \
    _rettype ucm_override_##_name(UCM_FUNC_DEFINE_ARGS(__VA_ARGS__)) \
    { \
        ucm_trace("%s()", __FUNCTION__); \
        \
        if (ucs_unlikely(ucm_cudamem_get_orig_thread == pthread_self())) { \
            return _fail_val; \
        } \
        return ucm_##_name(UCM_FUNC_PASS_ARGS(__VA_ARGS__)); \
    }

#define UCM_OVERRIDE_CUDA_FUNC(_name) \
    cudaError_t _name() __attribute__ ((alias ("ucm_override_" UCS_PP_QUOTE(_name)))); \


/*
 * Define argument list with given types.
 */
#define UCM_FUNC_DEFINE_ARGS(...) \
    UCS_PP_FOREACH_SEP(_UCM_FUNC_ARG_DEFINE, _, \
                       UCS_PP_ZIP((UCS_PP_SEQ(UCS_PP_NUM_ARGS(__VA_ARGS__))), \
                                  (__VA_ARGS__)))

/*
 * Pass auto-generated arguments to a function call.
 */
#define UCM_FUNC_PASS_ARGS(...) \
    UCS_PP_FOREACH_SEP(_UCM_FUNC_ARG_PASS, _, UCS_PP_SEQ(UCS_PP_NUM_ARGS(__VA_ARGS__)))


/*
 * Helpers
 */
#define _UCM_FUNC_ARG_DEFINE(_, _bundle) \
    __UCM_FUNC_ARG_DEFINE(_, UCS_PP_TUPLE_0 _bundle, UCS_PP_TUPLE_1 _bundle)
#define __UCM_FUNC_ARG_DEFINE(_, _index, _type) \
    _type UCS_PP_TOKENPASTE(arg, _index)
#define _UCM_FUNC_ARG_PASS(_, _index) \
    UCS_PP_TOKENPASTE(arg, _index)


UCM_DEFINE_CUDA_FUNC(cudaFree,   cudaError_t,  -1, void*)

#if ENABLE_SYMBOL_OVERRIDE
UCM_OVERRIDE_CUDA_FUNC(cudaFree)
#endif

/* =========================================================================
 * Copyright (c) 2025 AutoPoly Contributors
 * SPDX-License-Identifier: Apache-2.0
 * ========================================================================== */

/* isl_config.h.cmake - Configuration template for ISL */

/* define if your compiler has __attribute__ */
#cmakedefine HAVE___ATTRIBUTE__ /**/

/* most gcc compilers know a function __attribute__((__warn_unused_result__)) */
#define GCC_WARN_UNUSED_RESULT @GCC_WARN_UNUSED_RESULT@


/* Define to 1 if you have the declaration of `ffs', and to 0 if you don't. */
#define HAVE_DECL_FFS @HAVE_DECL_FFS@

/* Define to 1 if you have the declaration of `__builtin_ffs', and to 0 if you
   don't. */
#define HAVE_DECL___BUILTIN_FFS @HAVE_DECL___BUILTIN_FFS@

/* Define to 1 if you have the declaration of `_BitScanForward', and to 0 if
   you don't. */
#define HAVE_DECL__BITSCANFORWARD @HAVE_DECL__BITSCANFORWARD@


/* Define to 1 if you have the declaration of `strcasecmp', and to 0 if you
   don't. */
#define HAVE_DECL_STRCASECMP @HAVE_DECL_STRCASECMP@

/* Define to 1 if you have the declaration of `_stricmp', and to 0 if you
   don't. */
#define HAVE_DECL__STRICMP @HAVE_DECL__STRICMP@


/* Define to 1 if you have the declaration of `strncasecmp', and to 0 if you
   don't. */
#define HAVE_DECL_STRNCASECMP @HAVE_DECL_STRNCASECMP@

/* Define to 1 if you have the declaration of `_strnicmp', and to 0 if you
   don't. */
#define HAVE_DECL__STRNICMP @HAVE_DECL__STRNICMP@


/* Define to 1 if you have the declaration of `snprintf', and to 0 if you
   don't. */
#define HAVE_DECL_SNPRINTF @HAVE_DECL_SNPRINTF@

/* Define to 1 if you have the declaration of `_snprintf', and to 0 if you
   don't. */
#define HAVE_DECL__SNPRINTF @HAVE_DECL__SNPRINTF@


/* use gmp to implement isl_int */
#cmakedefine USE_GMP_FOR_MP

/* use imath to implement isl_int */
#cmakedefine USE_IMATH_FOR_MP

/* Use small integer optimization */
#cmakedefine USE_SMALL_INT_OPT

/* isl_config_post.h is the below context */
#ifndef HAVE___ATTRIBUTE__
#define __attribute__(x)
#endif

#if HAVE_DECL_FFS
#include <strings.h>
#endif

#if (HAVE_DECL_FFS==0) && (HAVE_DECL___BUILTIN_FFS==1)
#define ffs __builtin_ffs
#endif

#if !HAVE_DECL_FFS && !HAVE_DECL___BUILTIN_FFS && HAVE_DECL__BITSCANFORWARD
int isl_ffs(int i);
#define ffs isl_ffs
#endif

#if HAVE_DECL_STRCASECMP || HAVE_DECL_STRNCASECMP
#include <strings.h>
#endif

#if !HAVE_DECL_STRCASECMP && HAVE_DECL__STRICMP
#define strcasecmp _stricmp
#endif

#if !HAVE_DECL_STRNCASECMP && HAVE_DECL__STRNICMP
#define strncasecmp _strnicmp
#endif

#if HAVE_DECL__SNPRINTF
#define snprintf _snprintf
#endif

#ifdef GCC_WARN_UNUSED_RESULT
#define WARN_UNUSED	GCC_WARN_UNUSED_RESULT
#else
#define WARN_UNUSED
#endif
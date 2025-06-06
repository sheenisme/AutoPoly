//===- ppcg_wrapper.h - PPCG Wrapper Interface ----------------------------===//
//
// Copyright (c) 2025 AutoPoly Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file declares the interface to the PPCG wrapper for C code optimization.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOPOLY_PPCG_WRAPPER_H
#define AUTOPOLY_PPCG_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

/// Main entry point for PPCG C code optimization
/// This is a wrapper around the original PPCG main function
int ppcg_main(int argc, char *argv[]);

#ifdef __cplusplus
}
#endif

#endif // AUTOPOLY_PPCG_WRAPPER_H

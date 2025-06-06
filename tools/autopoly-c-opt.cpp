//===- autopoly-c-opt.cpp - AutoPoly C Optimizer --------------------------===//
//
// Copyright (c) 2025 AutoPoly Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file implements the main driver for the AutoPoly C optimizer tool,
// which applies polyhedral scheduling transformations to C code.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <string>
#include <cstdio>

#include "AutoPoly/ppcg_wrapper.h"

int main(int argc, char *argv[]) {
    ppcg_main(argc, argv);
    return 0;
}
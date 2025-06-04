//===- autopoly-mlir-opt.cpp - AutoPoly MLIR Optimizer --------*- C++ -*-===//
//
// Part of the AutoPoly Project
//
// This file implements the main driver for the AutoPoly MLIR optimizer tool,
// which applies polyhedral scheduling transformations to MLIR affine
// dialect operations.
//
//===----------------------------------------------------------------------===//

#include "AutoPoly/Passes/AutoPolyPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

using namespace mlir;
using namespace autopoly;

int main(int argc, char **argv) {
  // Register all MLIR dialects
  DialectRegistry registry;
  registerAllDialects(registry);
  
  // Register all MLIR passes
  registerAllPasses();
  
  // Register AutoPoly passes
  passes::registerAutoPolyPasses();
  
  return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "AutoPoly MLIR optimizer\n", registry));
}

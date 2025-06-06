//===- PolyhedralUtils.cpp - Polyhedral Analysis Utilities ----------------===//
//
// Copyright (c) 2025 AutoPoly Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file implements utility functions for polyhedral analysis and
// transformation in the AutoPoly framework.
//
//===----------------------------------------------------------------------===//

#include "AutoPoly/Analysis/PolyhedralExtraction.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/IR/Region.h"
#include "llvm/Support/Debug.h"

#include <isl/ctx.h>
#include <isl/options.h>

#define DEBUG_TYPE "polyhedral-utils"

using namespace mlir;

namespace autopoly {
namespace analysis {

// PolyhedralUtils implementation
isl_ctx* PolyhedralUtils::createContext() {
  isl_ctx* ctx = isl_ctx_alloc();
  
  if (!ctx) {
    llvm::errs() << "Failed to create ISL context\n";
    return nullptr;
  }
  
  // Set ISL options for better performance and debugging
  isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);
  
  LLVM_DEBUG(llvm::dbgs() << "Created ISL context\n");
  return ctx;
}

void PolyhedralUtils::destroyContext(isl_ctx* ctx) {
  if (ctx) {
    isl_ctx_free(ctx);
    LLVM_DEBUG(llvm::dbgs() << "Destroyed ISL context\n");
  }
}

bool PolyhedralUtils::isRegionAffine(Region& region) {
  // Check if all operations in the region are affine
  bool isAffine = true;
  
  region.walk([&](Operation* op) {
    // Check for affine operations
    if (isa<affine::AffineForOp, affine::AffineIfOp, affine::AffineLoadOp, 
            affine::AffineStoreOp, affine::AffineYieldOp>(op)) {
      return WalkResult::advance();
    }
    
    // Check for arithmetic operations (allowed in affine context)
    if (op->getDialect()->getNamespace() == "arith") {
      return WalkResult::advance();
    }
    
    // Check for func operations
    if (isa<func::FuncOp, func::ReturnOp>(op)) {
      return WalkResult::advance();
    }
    
    // If we reach here, it's not an affine operation
    LLVM_DEBUG(llvm::dbgs() << "Non-affine operation found: " << op->getName() << "\n");
    isAffine = false;
    return WalkResult::interrupt();
  });
  
  return isAffine;
}

bool PolyhedralUtils::hasAffineLoops(func::FuncOp funcOp) {
  bool hasLoops = false;
  
  funcOp.walk([&](affine::AffineForOp) {
    hasLoops = true;
    return WalkResult::interrupt();
  });
  
  return hasLoops;
}

int PolyhedralUtils::getLoopDepth(affine::AffineForOp forOp) {
  int depth = 1;
  Operation* parent = forOp->getParentOp();
  
  while (parent && !isa<func::FuncOp>(parent)) {
    if (isa<affine::AffineForOp>(parent)) {
      depth++;
    }
    parent = parent->getParentOp();
  }
  
  return depth;
}

std::vector<affine::AffineForOp> PolyhedralUtils::getNestedLoops(affine::AffineForOp forOp) {
  std::vector<affine::AffineForOp> loops;
  loops.push_back(forOp);
  
  forOp.walk([&](affine::AffineForOp nestedLoop) {
    if (nestedLoop != forOp) {
      loops.push_back(nestedLoop);
    }
  });
  
  return loops;
}

bool PolyhedralUtils::isPerfectLoopNest(affine::AffineForOp forOp) {
  // A perfect loop nest has only loop operations in the body (no other computations)
  bool isPerfect = true;
  
  forOp.getBody()->walk([&](Operation* op) {
    // Skip the terminator
    if (op->hasTrait<OpTrait::IsTerminator>()) {
      return WalkResult::advance();
    }
    
    // If we find any operation that's not a loop, it's not perfect
    if (!isa<affine::AffineForOp>(op)) {
      // Check if this is the innermost loop body
      bool hasNestedLoop = false;
      for (auto& nestedOp : op->getParentRegion()->getOps()) {
        if (isa<affine::AffineForOp>(nestedOp)) {
          hasNestedLoop = true;
          break;
        }
      }
      
      if (!hasNestedLoop) {
        // This is computation in the innermost loop, which is allowed
        return WalkResult::advance();
      } else {
        isPerfect = false;
        return WalkResult::interrupt();
      }
    }
    
    return WalkResult::advance();
  });
  
  return isPerfect;
}

std::vector<Operation*> PolyhedralUtils::getStatements(affine::AffineForOp forOp) {
  std::vector<Operation*> statements;
  
  forOp.getBody()->walk([&](Operation* op) {
    // Skip loop operations and terminators
    if (isa<affine::AffineForOp>(op) || op->hasTrait<OpTrait::IsTerminator>()) {
      return WalkResult::advance();
    }
    
    // Collect leaf operations (statements)
    if (isa<affine::AffineLoadOp, affine::AffineStoreOp>(op) ||
        op->getDialect()->getNamespace() == "arith") {
      statements.push_back(op);
    }
    
    return WalkResult::advance();
  });
  
  return statements;
}

bool PolyhedralUtils::hasComplexControl(func::FuncOp funcOp) {
  bool hasComplex = false;
  
  funcOp.walk([&](Operation* op) {
    // Check for complex control flow
    if (isa<affine::AffineIfOp>(op)) {
      // Affine if is acceptable
      return WalkResult::advance();
    }
    
    // Check for non-affine control flow
    if (op->getDialect()->getNamespace() == "scf" ||
        op->getDialect()->getNamespace() == "cf") {
      hasComplex = true;
      return WalkResult::interrupt();
    }
    
    return WalkResult::advance();
  });
  
  return hasComplex;
}

std::string PolyhedralUtils::getOperationSignature(Operation* op) {
  std::string signature;
  llvm::raw_string_ostream os(signature);
  
  os << op->getName().getStringRef();
  
  // Add operand types
  os << "(";
  for (auto operand : op->getOperands()) {
    os << operand.getType() << ",";
  }
  os << ")";
  
  // Add result types
  if (op->getNumResults() > 0) {
    os << " -> (";
    for (auto result : op->getResults()) {
      os << result.getType() << ",";
    }
    os << ")";
  }
  
  return os.str();
}

bool PolyhedralUtils::canExtractPolyhedralModel(func::FuncOp funcOp) {
  // Check basic requirements for polyhedral extraction
  
  // Must have affine loops
  if (!hasAffineLoops(funcOp)) {
    LLVM_DEBUG(llvm::dbgs() << "Function has no affine loops\n");
    return false;
  }
  
  // Must be purely affine
  if (!isRegionAffine(funcOp.getBody())) {
    LLVM_DEBUG(llvm::dbgs() << "Function contains non-affine operations\n");
    return false;
  }
  
  // Must not have complex control flow
  if (hasComplexControl(funcOp)) {
    LLVM_DEBUG(llvm::dbgs() << "Function has complex control flow\n");
    return false;
  }
  
  return true;
}

} // namespace analysis
} // namespace autopoly

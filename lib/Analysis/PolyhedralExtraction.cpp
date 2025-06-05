//===- PolyhedralExtraction.cpp - MLIR to Polyhedral Model ----*- C++ -*-===//
//
// Part of the AutoPoly Project
//
// This file implements the extraction of polyhedral model information from
// MLIR affine dialect operations, creating ISL schedule trees for
// scheduling transformation.
//
//===----------------------------------------------------------------------===//

#include "AutoPoly/Analysis/PolyhedralExtraction.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

// ISL headers
#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_set.h>
#include <isl/union_map.h>
#include <isl/schedule.h>
#include <isl/space.h>
#include <isl/schedule_node.h>
#include <isl/constraint.h>
#include <isl/options.h>

#define DEBUG_TYPE "polyhedral-extraction"

using namespace mlir;

using mlir::affine::AffineForOp;
using mlir::affine::AffineIfOp;
using mlir::affine::AffineLoadOp;
using mlir::affine::AffineStoreOp;
using mlir::affine::AffineYieldOp;

namespace autopoly {
namespace analysis {

// PolyhedralModel implementation
PolyhedralModel::PolyhedralModel(isl_ctx* ctx) : ctx_(ctx), schedule_(nullptr) {
  iteration_domain_ = isl_union_set_empty(isl_space_params_alloc(ctx_, 0));
}

PolyhedralModel::~PolyhedralModel() {
  if (schedule_) {
    isl_schedule_free(schedule_);
  }
  if (iteration_domain_) {
    isl_union_set_free(iteration_domain_);
  }
}

void PolyhedralModel::addStatement(const PolyhedralStatement& stmt) {
  statements_.push_back(stmt);
  
  // Add to iteration domain
  if (stmt.domain) {
    isl_set* domain_copy = isl_set_copy(stmt.domain);
    isl_union_set* domain_union = isl_union_set_from_set(domain_copy);
    iteration_domain_ = isl_union_set_union(iteration_domain_, domain_union);
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Added statement: " << stmt.name << "\n");
}

void PolyhedralModel::addArrayAccess(const ArrayAccessInfo& access) {
  array_accesses_.push_back(access);
  
  LLVM_DEBUG(llvm::dbgs() << "Added array access for: ");
  LLVM_DEBUG(llvm::dbgs() << access.array);
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

void PolyhedralModel::buildScheduleTree() {
  if (statements_.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No statements to build schedule from\n");
    return;
  }
  
  // Create initial identity schedule
  isl_union_map* identity_schedule = isl_union_map_empty(isl_space_params_alloc(ctx_, 0));
  
  for (size_t i = 0; i < statements_.size(); ++i) {
    const auto& stmt = statements_[i];
    if (!stmt.domain) continue;
    
    // Create identity map for this statement
    isl_space* space = isl_set_get_space(stmt.domain);
    int dim = isl_space_dim(space, isl_dim_set);
    
    isl_map* identity = isl_map_identity(isl_space_map_from_set(space));
    
    // Add statement dimension
    identity = isl_map_add_dims(identity, isl_dim_out, 1);
    identity = isl_map_fix_si(identity, isl_dim_out, dim, static_cast<int>(i));
    
    isl_union_map* stmt_schedule = isl_union_map_from_map(identity);
    identity_schedule = isl_union_map_union(identity_schedule, stmt_schedule);
  }
  
  // Convert to schedule
  schedule_ = isl_schedule_from_domain(isl_union_set_copy(iteration_domain_));
  schedule_ = isl_schedule_insert_partial_schedule(
      schedule_, isl_multi_union_pw_aff_from_union_map(identity_schedule));
  
  LLVM_DEBUG(llvm::dbgs() << "Built initial schedule tree\n");
}

bool PolyhedralModel::validate() const {
  if (!ctx_) {
    LLVM_DEBUG(llvm::dbgs() << "Error: No ISL context\n");
    return false;
  }
  
  if (statements_.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Warning: No statements in polyhedral model\n");
    return false;
  }
  
  if (!iteration_domain_) {
    LLVM_DEBUG(llvm::dbgs() << "Error: No iteration domain\n");
    return false;
  }
  
  // Check that all statements have valid domains
  for (const auto& stmt : statements_) {
    if (!stmt.domain) {
      LLVM_DEBUG(llvm::dbgs() << "Error: Statement " << stmt.name << " has no domain\n");
      return false;
    }
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Polyhedral model validation passed\n");
  return true;
}

// PolyhedralExtractor implementation
PolyhedralExtractor::PolyhedralExtractor(isl_ctx* ctx) : ctx_(ctx) {}

PolyhedralExtractor::~PolyhedralExtractor() = default;

std::unique_ptr<PolyhedralModel> PolyhedralExtractor::extractFromAffineFor(AffineForOp forOp) {
  auto model = std::make_unique<PolyhedralModel>(ctx_);
  
  loop_stack_.clear();
  value_to_name_.clear();
  loop_depth_.clear();
  statement_map_.clear();
  
  visitAffineFor(forOp, *model);
  
  model->buildScheduleTree();
  
  if (!model->validate()) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to extract valid polyhedral model\n");
    return nullptr;
  }
  
  return model;
}

std::unique_ptr<PolyhedralModel> PolyhedralExtractor::extractFromAffineNest(
    const std::vector<AffineForOp>& forOps) {
  
  auto model = std::make_unique<PolyhedralModel>(ctx_);
  
  loop_stack_.clear();
  value_to_name_.clear();
  loop_depth_.clear();
  statement_map_.clear();
  
  // Process the outermost loop
  if (!forOps.empty()) {
    visitAffineFor(forOps[0], *model);
  }
  
  model->buildScheduleTree();
  
  if (!model->validate()) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to extract valid polyhedral model from nest\n");
    return nullptr;
  }
  
  return model;
}

std::unique_ptr<PolyhedralModel> PolyhedralExtractor::extractFromRegion(Region& region) {
  auto model = std::make_unique<PolyhedralModel>(ctx_);
  
  loop_stack_.clear();
  value_to_name_.clear();
  loop_depth_.clear();
  statement_map_.clear();
  
  // Walk through all operations in the region
  region.walk([&](Operation* op) {
    visitOperation(op, *model);
  });
  
  model->buildScheduleTree();
  
  if (!model->validate()) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to extract valid polyhedral model from region\n");
    return nullptr;
  }
  
  return model;
}

std::unique_ptr<PolyhedralModel> PolyhedralExtractor::extractFromFunction(func::FuncOp funcOp) {
  auto model = std::make_unique<PolyhedralModel>(ctx_);
  
  loop_stack_.clear();
  value_to_name_.clear();
  loop_depth_.clear();
  statement_map_.clear();
  
  // Walk through the function body
  funcOp.walk([&](Operation* op) {
    visitOperation(op, *model);
  });
  
  model->buildScheduleTree();
  
  if (!model->validate()) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to extract valid polyhedral model from function\n");
    return nullptr;
  }
  
  return model;
}

void PolyhedralExtractor::visitOperation(Operation* op, PolyhedralModel& model) {
  if (auto forOp = dyn_cast<AffineForOp>(op)) {
    visitAffineFor(forOp, model);
  } else if (auto ifOp = dyn_cast<AffineIfOp>(op)) {
    visitAffineIf(ifOp, model);
  } else if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
    visitAffineLoad(loadOp, model);
  } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
    visitAffineStore(storeOp, model);
  } else if (auto yieldOp = dyn_cast<AffineYieldOp>(op)) {
    visitAffineYield(yieldOp, model);
  } else if (isAffineAccessPattern(op)) {
    // Handle other affine operations that access memory
    auto stmt = createStatement(op);
    model.addStatement(stmt);
    statement_map_[generateStatementName(op)] = &model.getStatements().back();
  }
}

void PolyhedralExtractor::visitAffineFor(AffineForOp forOp, PolyhedralModel& model) {
  // Add to loop stack
  loop_stack_.push_back(forOp);
  
  // Generate name for induction variable
  Value iv = forOp.getInductionVar();
  std::string iv_name = generateValueName(iv);
  value_to_name_[iv] = iv_name;
  loop_depth_[iv] = loop_stack_.size() - 1;
  
  LLVM_DEBUG(llvm::dbgs() << "Visiting affine for loop with IV: " << iv_name << "\n");
  
  // Create domain for this loop
  isl_set* domain = convertAffineDomain(forOp);
  
  // Process body
  for (auto& op : forOp.getBody()->getOperations()) {
    visitOperation(&op, model);
  }
  
  // Remove from loop stack
  loop_stack_.pop_back();
  
  if (domain) {
    isl_set_free(domain);
  }
}

void PolyhedralExtractor::visitAffineIf(AffineIfOp ifOp, PolyhedralModel& model) {
  LLVM_DEBUG(llvm::dbgs() << "Visiting affine if operation\n");
  
  // Process then block
  if (ifOp.getThenBlock()) {
    for (auto& op : ifOp.getThenBlock()->getOperations()) {
      visitOperation(&op, model);
    }
  }
  
  // Process else block if present
  if (ifOp.getElseBlock()) {
    for (auto& op : ifOp.getElseBlock()->getOperations()) {
      visitOperation(&op, model);
    }
  }
}

void PolyhedralExtractor::visitAffineLoad(AffineLoadOp loadOp, PolyhedralModel& model) {
  LLVM_DEBUG(llvm::dbgs() << "Visiting affine load operation\n");
  
  // Create array access info
  Value array = loadOp.getMemRef();
  AffineMap accessMap = loadOp.getAffineMap();
  
  auto access = createArrayAccess(array, accessMap, true, loadOp.getOperation());
  model.addArrayAccess(access);
  
  // Create statement for this load
  auto stmt = createStatement(loadOp.getOperation());
  model.addStatement(stmt);
  statement_map_[stmt.name] = &model.getStatements().back();
}

void PolyhedralExtractor::visitAffineStore(AffineStoreOp storeOp, PolyhedralModel& model) {
  LLVM_DEBUG(llvm::dbgs() << "Visiting affine store operation\n");
  
  // Create array access info
  Value array = storeOp.getMemRef();
  AffineMap accessMap = storeOp.getAffineMap();
  
  auto access = createArrayAccess(array, accessMap, false, storeOp.getOperation());
  model.addArrayAccess(access);
  
  // Create statement for this store
  auto stmt = createStatement(storeOp.getOperation());
  model.addStatement(stmt);
  statement_map_[stmt.name] = &model.getStatements().back();
}

void PolyhedralExtractor::visitAffineYield(AffineYieldOp yieldOp, PolyhedralModel& model) {
  LLVM_DEBUG(llvm::dbgs() << "Visiting affine yield operation\n");
  
  // Create statement for yield (needed for reduction patterns)
  auto stmt = createStatement(yieldOp.getOperation());
  model.addStatement(stmt);
  statement_map_[stmt.name] = &model.getStatements().back();
}

isl_set* PolyhedralExtractor::convertAffineDomain(AffineForOp forOp) {
  // Create space for the loop
  isl_space* space = isl_space_set_alloc(ctx_, 0, loop_stack_.size());
  
  // Set dimension names
  for (size_t i = 0; i < loop_stack_.size(); ++i) {
    std::string dim_name = "i" + std::to_string(i);
    space = isl_space_set_dim_name(space, isl_dim_set, i, dim_name.c_str());
  }
  
  // Create basic set
  isl_basic_set* bset = isl_basic_set_universe(isl_space_copy(space));
  
  // Add loop bounds constraints
  int current_dim = loop_stack_.size() - 1;
  
  // Lower bound: iv >= lb
  isl_constraint* lb_constraint = isl_constraint_alloc_inequality(
      isl_local_space_from_space(isl_space_copy(space)));
  lb_constraint = isl_constraint_set_coefficient_si(lb_constraint, isl_dim_set, current_dim, 1);
  
  // For simplicity, assume constant bounds (extend for affine bounds)
  if (auto lb_const = forOp.getConstantLowerBound()) {
    lb_constraint = isl_constraint_set_constant_si(lb_constraint, -lb_const);
  }
  
  bset = isl_basic_set_add_constraint(bset, lb_constraint);
  
  // Upper bound: iv < ub
  isl_constraint* ub_constraint = isl_constraint_alloc_inequality(
      isl_local_space_from_space(isl_space_copy(space)));
  ub_constraint = isl_constraint_set_coefficient_si(ub_constraint, isl_dim_set, current_dim, -1);
  
  if (auto ub_const = forOp.getConstantUpperBound()) {
    ub_constraint = isl_constraint_set_constant_si(ub_constraint, ub_const - 1);
  }
  
  bset = isl_basic_set_add_constraint(bset, ub_constraint);
  
  isl_space_free(space);
  return isl_set_from_basic_set(bset);
}

PolyhedralStatement PolyhedralExtractor::createStatement(Operation* op) {
  PolyhedralStatement stmt;
  stmt.operation = op;
  stmt.name = generateStatementName(op);
  
  // Create domain based on current loop stack
  if (!loop_stack_.empty()) {
    stmt.domain = convertAffineDomain(loop_stack_.back());
  } else {
    // Scalar statement
    isl_space* space = isl_space_set_alloc(ctx_, 0, 0);
    stmt.domain = isl_set_universe(space);
  }
  
  // Analyze access patterns
  if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
    stmt.accessed_arrays.push_back(loadOp.getMemRef());
    stmt.is_read_only = true;
  } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
    stmt.accessed_arrays.push_back(storeOp.getMemRef());
    stmt.is_write_only = true;
  }
  
  // Estimate computational intensity
  stmt.computational_intensity = 1; // Simple estimate
  
  return stmt;
}

ArrayAccessInfo PolyhedralExtractor::createArrayAccess(Value array, AffineMap accessMap,
                                                      bool isRead, Operation* context) {
  ArrayAccessInfo access;
  access.array = array;
  access.array_type = array.getType().cast<MemRefType>();
  access.is_affine = true; // AffineMap guarantees this
  
  // Convert access map to ISL
  isl_map* isl_access_map = convertAffineMap(accessMap, {});
  
  if (isRead) {
    access.read_accesses.push_back(isl_access_map);
  } else {
    access.write_accesses.push_back(isl_access_map);
  }
  
  // Analyze access pattern properties
  access.stride_patterns = analyzeStridePattern(accessMap);
  access.has_temporal_reuse = hasTemporalReuse(access.read_accesses);
  access.has_spatial_reuse = hasSpatialReuse(access.read_accesses);
  
  return access;
}

isl_map* PolyhedralExtractor::convertAffineMap(AffineMap affineMap, 
                                              const std::vector<Value>& operands) {
  // Simplified conversion - in practice would need full AffineMap to ISL conversion
  int domain_dims = loop_stack_.size();
  int range_dims = affineMap.getNumResults();
  
  isl_space* space = isl_space_alloc(ctx_, 0, domain_dims, range_dims);
  isl_map* map = isl_map_universe(space);
  
  // This is a simplified implementation
  // Full implementation would parse AffineMap expressions
  
  return map;
}

std::string PolyhedralExtractor::generateStatementName(Operation* op) {
  static int counter = 0;
  std::string base = "S";
  
  if (isa<AffineLoadOp>(op)) {
    base = "Load";
  } else if (isa<AffineStoreOp>(op)) {
    base = "Store";
  } else if (isa<AffineYieldOp>(op)) {
    base = "Yield";
  }
  
  return base + std::to_string(counter++);
}

std::string PolyhedralExtractor::generateArrayName(Value array) {
  // Try to get name from operation
  if (auto defOp = array.getDefiningOp()) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(defOp)) {
      return "Array" + std::to_string(reinterpret_cast<uintptr_t>(allocOp.getOperation()));
    }
  }
  
  return "Array" + std::to_string(reinterpret_cast<uintptr_t>(array.getAsOpaquePointer()));
}

std::string PolyhedralExtractor::generateValueName(Value value) {
  return "v" + std::to_string(reinterpret_cast<uintptr_t>(value.getAsOpaquePointer()));
}

bool PolyhedralExtractor::isAffineAccessPattern(Operation* op) {
  // Check if operation involves affine memory accesses
  return isa<AffineLoadOp, AffineStoreOp, AffineYieldOp>(op);
}

std::vector<int> PolyhedralExtractor::analyzeStridePattern(AffineMap accessMap) {
  std::vector<int> strides;
  
  // Simplified stride analysis
  for (unsigned i = 0; i < accessMap.getNumResults(); ++i) {
    // Assume unit stride for now
    strides.push_back(1);
  }
  
  return strides;
}

bool PolyhedralExtractor::hasTemporalReuse(const std::vector<isl_map*>& accesses) {
  // Simplified reuse analysis
  return accesses.size() > 1;
}

bool PolyhedralExtractor::hasSpatialReuse(const std::vector<isl_map*>& accesses) {
  // Simplified reuse analysis
  return true; // Assume spatial reuse exists
}

} // namespace analysis
} // namespace autopoly

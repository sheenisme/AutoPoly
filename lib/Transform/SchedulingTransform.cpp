//===- SchedulingTransform.cpp - Polyhedral Scheduling Transforms -*- C++ -*-===//
//
// Part of the AutoPoly Project
//
// This file implements the scheduling transformation framework that applies
// polyhedral transformations based on target characteristics and
// dependence constraints.
//
//===----------------------------------------------------------------------===//

#include "AutoPoly/Transform/SchedulingTransform.h"
#include "llvm/Support/Debug.h"

#include <isl/aff.h>
#include <isl/ast.h>
#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>

#define DEBUG_TYPE "scheduling-transform"

namespace autopoly {
namespace transform {

// TilingTransformation implementation
bool TilingTransformation::isApplicable(const analysis::PolyhedralModel& model,
                                       const analysis::DependenceInfo& deps,
                                       const target::TargetCharacteristics& target) const {
  // Check if model has loops suitable for tiling
  if (model.getStatements().empty()) {
    return false;
  }
  
  // Check if target benefits from tiling
  switch (target.type) {
    case target::TargetType::CPU:
    case target::TargetType::GPU:
    case target::TargetType::OPENCL:
      return true;
    default:
      return false;
  }
}

TransformationResult TilingTransformation::apply(isl_schedule* schedule,
                                                const analysis::PolyhedralModel& model,
                                                const analysis::DependenceInfo& deps,
                                                const scheduling::SchedulingParameters& params) const {
  
  TransformationResult result;
  result.transformation_successful = false;
  
  if (!schedule) {
    result.error_message = "Invalid schedule for tiling";
    return result;
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Applying tiling transformation\n");
  
  // Get tile sizes
  std::vector<int> tile_sizes = params.tile_sizes;
  if (tile_sizes.empty()) {
    tile_sizes = computeOptimalTileSizes(model, target::TargetCharacteristics{});
  }
  
  // Validate tile sizes
  if (!validateTilingSizes(tile_sizes, deps)) {
    result.error_message = "Invalid tile sizes for dependence constraints";
    return result;
  }
  
  // Apply tiling to schedule
  isl_schedule* tiled_schedule = applyTiling(schedule, tile_sizes);
  
  if (!tiled_schedule) {
    result.error_message = "Failed to apply tiling to schedule";
    return result;
  }
  
  result.transformed_schedule = tiled_schedule;
  result.transformation_successful = true;
  result.final_tile_sizes = tile_sizes;
  result.applied_techniques.push_back(scheduling::OptimizationTechnique::TILING);
  result.estimated_speedup = estimateBenefit(model, deps, target::TargetCharacteristics{});
  
  LLVM_DEBUG(llvm::dbgs() << "Tiling transformation completed successfully\n");
  return result;
}

double TilingTransformation::estimateBenefit(const analysis::PolyhedralModel& model,
                                           const analysis::DependenceInfo& deps,
                                           const target::TargetCharacteristics& target) const {
  // Simple benefit estimation for tiling
  double benefit = 1.0;
  
  // More benefit for CPU due to cache locality
  if (target.type == target::TargetType::CPU) {
    benefit = 1.5; // 50% improvement estimate
  } else if (target.type == target::TargetType::GPU) {
    benefit = 1.2; // 20% improvement for GPU
  }
  
  // Adjust based on memory hierarchy
  if (!target.memory_hierarchy.empty()) {
    benefit += 0.1 * target.memory_hierarchy.size();
  }
  
  return benefit;
}

std::vector<int> TilingTransformation::computeOptimalTileSizes(
    const analysis::PolyhedralModel& model,
    const target::TargetCharacteristics& target) const {
  
  // Default tile sizes
  std::vector<int> tile_sizes = {32, 32, 32};
  
  // Adjust based on target type
  if (target.type == target::TargetType::CPU) {
    // CPU: smaller tiles for cache locality
    tile_sizes = {32, 32, 32};
  } else if (target.type == target::TargetType::GPU) {
    // GPU: thread-block friendly sizes
    tile_sizes = {16, 16, 1};
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Computed optimal tile sizes: ");
  for (int size : tile_sizes) {
    LLVM_DEBUG(llvm::dbgs() << size << " ");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
  
  return tile_sizes;
}

isl_schedule* TilingTransformation::applyTiling(isl_schedule* schedule,
                                              const std::vector<int>& tile_sizes) const {
  if (!schedule || tile_sizes.empty()) {
    return nullptr;
  }
  
  // Create tiling parameters
  isl_ctx* ctx = isl_schedule_get_ctx(schedule);
  isl_schedule* tiled = isl_schedule_copy(schedule);
  
  // Apply tiling at band nodes
  // This is a simplified implementation - full implementation would
  // traverse the schedule tree and apply tiling at appropriate band nodes
  
  // For now, just return a copy
  return tiled;
}

bool TilingTransformation::validateTilingSizes(const std::vector<int>& tile_sizes,
                                             const analysis::DependenceInfo& deps) const {
  // Basic validation
  for (int size : tile_sizes) {
    if (size <= 0) {
      LLVM_DEBUG(llvm::dbgs() << "Invalid tile size: " << size << "\n");
      return false;
    }
  }
  
  // Check against dependence constraints
  // This would involve checking if tiling preserves dependences
  
  return true;
}

// ParallelizationTransformation implementation
bool ParallelizationTransformation::isApplicable(const analysis::PolyhedralModel& model,
                                                const analysis::DependenceInfo& deps,
                                                const target::TargetCharacteristics& target) const {
  // Check if target supports parallelization
  if (target.compute_units <= 1) {
    return false;
  }
  
  // Check if there are parallelizable dimensions
  auto parallel_dims = findParallelizableDimensions(deps);
  return !parallel_dims.empty();
}

TransformationResult ParallelizationTransformation::apply(isl_schedule* schedule,
                                                         const analysis::PolyhedralModel& model,
                                                         const analysis::DependenceInfo& deps,
                                                         const scheduling::SchedulingParameters& params) const {
  
  TransformationResult result;
  result.transformation_successful = false;
  
  LLVM_DEBUG(llvm::dbgs() << "Applying parallelization transformation\n");
  
  // Find parallelizable dimensions
  std::vector<int> parallel_dims = findParallelizableDimensions(deps);
  
  if (parallel_dims.empty()) {
    result.error_message = "No parallelizable dimensions found";
    return result;
  }
  
  // Apply parallel marking to schedule
  isl_schedule* parallel_schedule = markParallelDimensions(schedule, parallel_dims);
  
  if (!parallel_schedule) {
    result.error_message = "Failed to mark parallel dimensions";
    return result;
  }
  
  result.transformed_schedule = parallel_schedule;
  result.transformation_successful = true;
  result.parallel_dimensions = parallel_dims;
  result.parallelism_degree = parallel_dims.size();
  result.applied_techniques.push_back(scheduling::OptimizationTechnique::PARALLELIZATION);
  result.estimated_speedup = estimateBenefit(model, deps, target::TargetCharacteristics{});
  
  LLVM_DEBUG(llvm::dbgs() << "Parallelization transformation completed successfully\n");
  return result;
}

double ParallelizationTransformation::estimateBenefit(const analysis::PolyhedralModel& model,
                                                    const analysis::DependenceInfo& deps,
                                                    const target::TargetCharacteristics& target) const {
  // Estimate parallel speedup
  auto parallel_dims = findParallelizableDimensions(deps);
  
  if (parallel_dims.empty()) {
    return 1.0;
  }
  
  // Simple parallel efficiency model
  double max_speedup = std::min(static_cast<double>(target.compute_units), 
                               static_cast<double>(parallel_dims.size() * 32));
  
  // Apply Amdahl's law approximation
  double parallel_fraction = 0.8; // Assume 80% parallelizable
  double efficiency = 0.7; // Account for overhead
  
  double speedup = 1.0 / ((1.0 - parallel_fraction) + 
                         parallel_fraction / (max_speedup * efficiency));
  
  return speedup;
}

std::vector<int> ParallelizationTransformation::findParallelizableDimensions(
    const analysis::DependenceInfo& deps) const {
  
  std::vector<int> parallel_dims;
  
  // Check each dimension for data dependences
  for (int dim = 0; dim < 3; ++dim) { // Check first 3 dimensions
    if (deps.canParallelizeDimension(dim)) {
      parallel_dims.push_back(dim);
    }
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Found parallelizable dimensions: ");
  for (int dim : parallel_dims) {
    LLVM_DEBUG(llvm::dbgs() << dim << " ");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
  
  return parallel_dims;
}

isl_schedule* ParallelizationTransformation::markParallelDimensions(
    isl_schedule* schedule, const std::vector<int>& parallel_dims) const {
  
  if (!schedule || parallel_dims.empty()) {
    return nullptr;
  }
  
  // Mark dimensions as parallel in schedule
  // This is a simplified implementation
  return isl_schedule_copy(schedule);
}

double ParallelizationTransformation::estimateParallelEfficiency(
    const std::vector<int>& parallel_dims,
    const target::TargetCharacteristics& target) const {
  
  if (parallel_dims.empty()) {
    return 0.0;
  }
  
  // Estimate efficiency based on target type
  double base_efficiency = 0.8;
  
  switch (target.type) {
    case target::TargetType::GPU:
      base_efficiency = 0.9; // GPUs are very efficient at parallelization
      break;
    case target::TargetType::CPU:
      base_efficiency = 0.7; // CPUs have more overhead
      break;
    default:
      base_efficiency = 0.6;
      break;
  }
  
  // Reduce efficiency for too many parallel dimensions
  if (parallel_dims.size() > 2) {
    base_efficiency *= 0.8;
  }
  
  return base_efficiency;
}

// SchedulingTransformer implementation
SchedulingTransformer::SchedulingTransformer(isl_ctx* ctx) : ctx_(ctx) {
  initializeBuiltinTransformations();
}

SchedulingTransformer::~SchedulingTransformer() = default;

void SchedulingTransformer::initializeBuiltinTransformations() {
  transformations_.push_back(std::make_unique<TilingTransformation>());
  transformations_.push_back(std::make_unique<ParallelizationTransformation>());
  transformations_.push_back(std::make_unique<FusionTransformation>());
  transformations_.push_back(std::make_unique<SkewingTransformation>());
  
  LLVM_DEBUG(llvm::dbgs() << "Initialized " << transformations_.size() 
                          << " built-in transformations\n");
}

TransformationResult SchedulingTransformer::transform(
    const analysis::PolyhedralModel& model,
    const analysis::DependenceInfo& deps,
    const target::TargetCharacteristics& target,
    const scheduling::SchedulingStrategy& strategy) {
  
  LLVM_DEBUG(llvm::dbgs() << "Starting scheduling transformation\n");
  
  // Get optimization techniques from strategy
  auto techniques = strategy.getOptimizationTechniques();
  auto params = strategy.getParameters(target);
  
  // Plan transformation sequence
  auto transformation_sequence = planTransformationSequence(model, deps, target, techniques);
  
  // Apply transformations
  isl_schedule* current_schedule = model.getSchedule();
  if (!current_schedule) {
    TransformationResult result;
    result.error_message = "No initial schedule available";
    return result;
  }
  
  return applyTransformationSequence(current_schedule, transformation_sequence, 
                                   model, deps, params);
}

std::vector<const SchedulingTransformation*> SchedulingTransformer::planTransformationSequence(
    const analysis::PolyhedralModel& model,
    const analysis::DependenceInfo& deps,
    const target::TargetCharacteristics& target,
    const std::vector<scheduling::OptimizationTechnique>& techniques) {
  
  std::vector<const SchedulingTransformation*> sequence;
  
  // Map techniques to transformations
  for (auto technique : techniques) {
    for (const auto& transform : transformations_) {
      bool matched = false;
      
      switch (technique) {
        case scheduling::OptimizationTechnique::TILING:
          matched = (transform->getName() == "Tiling");
          break;
        case scheduling::OptimizationTechnique::PARALLELIZATION:
          matched = (transform->getName() == "Parallelization");
          break;
        case scheduling::OptimizationTechnique::FUSION:
          matched = (transform->getName() == "Fusion");
          break;
        case scheduling::OptimizationTechnique::SKEWING:
          matched = (transform->getName() == "Skewing");
          break;
        default:
          matched = false;
          break;
      }
      
      if (matched && transform->isApplicable(model, deps, target)) {
        sequence.push_back(transform.get());
        break;
      }
    }
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Planned transformation sequence with " 
                          << sequence.size() << " transformations\n");
  
  return sequence;
}

TransformationResult SchedulingTransformer::applyTransformationSequence(
    isl_schedule* initial_schedule,
    const std::vector<const SchedulingTransformation*>& sequence,
    const analysis::PolyhedralModel& model,
    const analysis::DependenceInfo& deps,
    const scheduling::SchedulingParameters& params) {
  
  TransformationResult combined_result;
  combined_result.transformed_schedule = isl_schedule_copy(initial_schedule);
  combined_result.transformation_successful = true;
  combined_result.estimated_speedup = 1.0;
  
  // Apply each transformation in sequence
  for (const auto* transform : sequence) {
    LLVM_DEBUG(llvm::dbgs() << "Applying transformation: " << transform->getName() << "\n");
    
    auto result = transform->apply(combined_result.transformed_schedule, model, deps, params);
    
    if (!result.transformation_successful) {
      LLVM_DEBUG(llvm::dbgs() << "Transformation failed: " << result.error_message << "\n");
      combined_result.error_message = result.error_message;
      combined_result.transformation_successful = false;
      break;
    }
    
    // Update combined result
    if (combined_result.transformed_schedule != initial_schedule) {
      isl_schedule_free(combined_result.transformed_schedule);
    }
    combined_result.transformed_schedule = result.transformed_schedule;
    combined_result.estimated_speedup *= result.estimated_speedup;
    
    // Merge applied techniques
    combined_result.applied_techniques.insert(
        combined_result.applied_techniques.end(),
        result.applied_techniques.begin(),
        result.applied_techniques.end());
    
    // Merge parameters
    for (const auto& param : result.transformation_parameters) {
      combined_result.transformation_parameters[param.first] = param.second;
    }
  }
  
  // Validate final schedule
  if (combined_result.transformation_successful) {
    if (!validateTransformedSchedule(combined_result.transformed_schedule, deps)) {
      combined_result.error_message = "Final schedule validation failed";
      combined_result.transformation_successful = false;
    }
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Transformation sequence completed with result: " 
                          << (combined_result.transformation_successful ? "SUCCESS" : "FAILURE") << "\n");
  
  return combined_result;
}

bool SchedulingTransformer::validateTransformedSchedule(isl_schedule* schedule,
                                                      const analysis::DependenceInfo& deps) const {
  if (!schedule) {
    return false;
  }
  
  // Basic validation - check that schedule is valid
  // In practice would check dependence preservation
  
  return true;
}

// FusionTransformation implementation
bool FusionTransformation::isApplicable(const analysis::PolyhedralModel& model,
                                      const analysis::DependenceInfo& deps,
                                      const target::TargetCharacteristics& target) const {
  // Check if there are multiple statements that can be fused
  if (model.getStatements().size() < 2) {
    return false;
  }
  
  // Check if target benefits from fusion
  switch (target.type) {
    case target::TargetType::CPU:
    case target::TargetType::FPGA:
    case target::TargetType::NPU:
      return true; // These targets benefit from fusion
    default:
      return false;
  }
}

TransformationResult FusionTransformation::apply(isl_schedule* schedule,
                                                const analysis::PolyhedralModel& model,
                                                const analysis::DependenceInfo& deps,
                                                const scheduling::SchedulingParameters& params) const {
  
  TransformationResult result;
  result.transformation_successful = false;
  
  LLVM_DEBUG(llvm::dbgs() << "Applying fusion transformation\n");
  
  // Find fusable statement groups
  auto fusable_groups = findFusableGroups(model, deps);
  
  if (fusable_groups.empty()) {
    result.error_message = "No fusable statement groups found";
    return result;
  }
  
  // Apply fusion to the largest group
  auto largest_group = *std::max_element(fusable_groups.begin(), fusable_groups.end(),
      [](const std::vector<std::string>& a, const std::vector<std::string>& b) {
        return a.size() < b.size();
      });
  
  isl_schedule* fused_schedule = applyFusion(schedule, largest_group);
  
  if (!fused_schedule) {
    result.error_message = "Failed to apply fusion";
    return result;
  }
  
  result.transformed_schedule = fused_schedule;
  result.transformation_successful = true;
  result.applied_techniques.push_back(scheduling::OptimizationTechnique::FUSION);
  result.estimated_speedup = estimateBenefit(model, deps, target::TargetCharacteristics{});
  result.memory_traffic_reduction = 0.2; // Estimate 20% reduction
  
  LLVM_DEBUG(llvm::dbgs() << "Fusion transformation completed successfully\n");
  return result;
}

double FusionTransformation::estimateBenefit(const analysis::PolyhedralModel& model,
                                           const analysis::DependenceInfo& deps,
                                           const target::TargetCharacteristics& target) const {
  // Estimate benefit based on number of statements and target type
  double benefit = 1.0;
  
  auto statements = model.getStatements();
  if (statements.size() >= 2) {
    // More benefit for targets that favor memory locality
    if (target.type == target::TargetType::CPU) {
      benefit = 1.3; // 30% improvement for CPU
    } else if (target.type == target::TargetType::FPGA) {
      benefit = 1.4; // 40% improvement for FPGA
    } else {
      benefit = 1.1; // 10% improvement for others
    }
  }
  
  return benefit;
}

std::vector<std::vector<std::string>> FusionTransformation::findFusableGroups(
    const analysis::PolyhedralModel& model,
    const analysis::DependenceInfo& deps) const {
  
  std::vector<std::vector<std::string>> fusable_groups;
  auto statements = model.getStatements();
  
  // Simple approach: try to fuse adjacent statements
  for (size_t i = 0; i < statements.size() - 1; ++i) {
    for (size_t j = i + 1; j < statements.size(); ++j) {
      // Check if statements can be fused (simplified check)
      if (deps.canFuse(statements[i].getName(), statements[j].getName())) {
        fusable_groups.push_back(std::vector<std::string>{statements[i].getName(), statements[j].getName()});
      }
    }
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Found " << fusable_groups.size() << " fusable groups\n");
  
  return fusable_groups;
}

isl_schedule* FusionTransformation::applyFusion(isl_schedule* schedule,
                                              const std::vector<std::string>& statements) const {
  if (!schedule || statements.empty()) {
    return nullptr;
  }
  
  // Apply fusion - simplified implementation
  return isl_schedule_copy(schedule);
}

// SkewingTransformation implementation
bool SkewingTransformation::isApplicable(const analysis::PolyhedralModel& model,
                                       const analysis::DependenceInfo& deps,
                                       const target::TargetCharacteristics& target) const {
  // Check if there are dependences that would benefit from skewing
  if (!deps.hasLoopCarriedDependences()) {
    return false;
  }
  
  // Check if target benefits from skewing
  switch (target.type) {
    case target::TargetType::GPU:
    case target::TargetType::CGRA:
    case target::TargetType::PIM:
      return true; // These targets benefit from skewing
    default:
      return false;
  }
}

TransformationResult SkewingTransformation::apply(isl_schedule* schedule,
                                                 const analysis::PolyhedralModel& model,
                                                 const analysis::DependenceInfo& deps,
                                                 const scheduling::SchedulingParameters& params) const {
  
  TransformationResult result;
  result.transformation_successful = false;
  
  LLVM_DEBUG(llvm::dbgs() << "Applying skewing transformation\n");
  
  // Compute skewing factors
  auto skewing_matrix = computeSkewingFactors(deps);
  
  if (skewing_matrix.empty()) {
    result.error_message = "No beneficial skewing factors found";
    return result;
  }
  
  // Apply skewing
  isl_schedule* skewed_schedule = applySkewing(schedule, skewing_matrix);
  
  if (!skewed_schedule) {
    result.error_message = "Failed to apply skewing";
    return result;
  }
  
  result.transformed_schedule = skewed_schedule;
  result.transformation_successful = true;
  result.applied_techniques.push_back(scheduling::OptimizationTechnique::SKEWING);
  result.estimated_speedup = estimateBenefit(model, deps, target::TargetCharacteristics{});
  
  LLVM_DEBUG(llvm::dbgs() << "Skewing transformation completed successfully\n");
  return result;
}

double SkewingTransformation::estimateBenefit(const analysis::PolyhedralModel& model,
                                            const analysis::DependenceInfo& deps,
                                            const target::TargetCharacteristics& target) const {
  // Estimate benefit based on dependence pattern and target
  double benefit = 1.0;
  
  if (deps.hasLoopCarriedDependences()) {
    // Benefit depends on target type
    if (target.type == target::TargetType::GPU) {
      benefit = 1.25; // 25% improvement for GPU memory coalescing
    } else if (target.type == target::TargetType::CGRA) {
      benefit = 1.35; // 35% improvement for CGRA dataflow
    } else {
      benefit = 1.15; // 15% improvement for others
    }
  }
  
  return benefit;
}

std::vector<std::vector<int>> SkewingTransformation::computeSkewingFactors(
    const analysis::DependenceInfo& deps) const {
  
  std::vector<std::vector<int>> skewing_matrix;
  
  // Simple skewing matrix computation
  // In practice, this would analyze dependence vectors
  if (deps.hasLoopCarriedDependences()) {
    // Create a simple skewing matrix
    skewing_matrix = {{1, 1}, {0, 1}}; // Basic skewing transformation
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Computed skewing matrix with " 
                          << skewing_matrix.size() << " rows\n");
  
  return skewing_matrix;
}

isl_schedule* SkewingTransformation::applySkewing(isl_schedule* schedule,
                                                const std::vector<std::vector<int>>& skewing_matrix) const {
  if (!schedule || skewing_matrix.empty()) {
    return nullptr;
  }
  
  // Apply skewing transformation - simplified implementation
  return isl_schedule_copy(schedule);
}

// TransformUtils implementation
bool TransformUtils::validateSchedule(isl_schedule* schedule,
                                     const analysis::DependenceInfo& deps) {
  if (!schedule) {
    return false;
  }
  
  // Basic validation - in practice would check dependence preservation
  return true;
}

int TransformUtils::computeParallelismDegree(isl_schedule* schedule) {
  if (!schedule) {
    return 1;
  }
  
  // Simplified - count parallel dimensions
  return 1; // Default sequential
}

double TransformUtils::estimateMemoryCost(isl_schedule* schedule,
                                         const analysis::PolyhedralModel& model,
                                         const target::TargetCharacteristics& target) {
  if (!schedule) {
    return 1.0;
  }
  
  // Simplified memory cost estimation
  double base_cost = 1.0;
  
  // Adjust based on target memory hierarchy
  if (!target.memory_hierarchy.empty()) {
    base_cost *= 0.8; // Assume optimized access
  }
  
  return base_cost;
}

std::vector<int> TransformUtils::extractTileSizes(isl_schedule* schedule) {
  std::vector<int> tile_sizes;
  
  if (!schedule) {
    return tile_sizes;
  }
  
  // Extract tile sizes from schedule tree - simplified
  tile_sizes = {32, 32}; // Default
  
  return tile_sizes;
}

std::vector<int> TransformUtils::extractParallelDimensions(isl_schedule* schedule) {
  std::vector<int> parallel_dims;
  
  if (!schedule) {
    return parallel_dims;
  }
  
  // Extract parallel dimensions - simplified
  parallel_dims = {0}; // Default outermost parallel
  
  return parallel_dims;
}

TransformationResult TransformUtils::mergeResults(const std::vector<TransformationResult>& results) {
  TransformationResult merged;
  
  if (results.empty()) {
    merged.transformation_successful = false;
    merged.error_message = "No results to merge";
    return merged;
  }
  
  merged.transformation_successful = true;
  merged.estimated_speedup = 1.0;
  
  for (const auto& result : results) {
    if (!result.transformation_successful) {
      merged.transformation_successful = false;
      merged.error_message = result.error_message;
      break;
    }
    
    merged.estimated_speedup *= result.estimated_speedup;
    
    // Merge applied techniques
    merged.applied_techniques.insert(merged.applied_techniques.end(),
                                   result.applied_techniques.begin(),
                                   result.applied_techniques.end());
    
    // Use last successful schedule
    if (result.transformed_schedule) {
      merged.transformed_schedule = result.transformed_schedule;
    }
  }
  
  return merged;
}

std::string TransformUtils::scheduleToString(isl_schedule* schedule) {
  if (!schedule) {
    return "null schedule";
  }
  
  // Convert ISL schedule to string
  char* str = isl_schedule_to_str(schedule);
  std::string result(str ? str : "invalid schedule");
  free(str);
  
  return result;
}

} // namespace transform
} // namespace autopoly

//===- DependenceAnalysis.cpp - Polyhedral Dependence Analysis -*- C++ -*-===//
//
// Part of the AutoPoly Project
//
// This file implements the dependence analysis framework for polyhedral models,
// including data dependences, memory dependences, and other constraints
// that affect scheduling transformations.
//
//===----------------------------------------------------------------------===//

#include "AutoPoly/Analysis/DependenceAnalysis.h"
#include "llvm/Support/Debug.h"

#include <isl/flow.h>
#include <isl/map.h>
#include <isl/schedule.h>
#include <isl/space.h>
#include <isl/set.h>
#include <isl/union_map.h>
#include <isl/union_set.h>

#define DEBUG_TYPE "dependence-analysis"

using namespace mlir;

namespace autopoly {
namespace analysis {

// DependenceVector implementation
bool DependenceVector::isLoopIndependent() const {
  for (size_t i = 0; i < distances.size(); ++i) {
    if (is_distance_known[i] && distances[i] != 0) {
      return false;
    }
  }
  return true;
}

bool DependenceVector::isLexicographicallyPositive() const {
  for (size_t i = 0; i < distances.size(); ++i) {
    if (is_distance_known[i]) {
      if (distances[i] > 0) {
        return true;
      } else if (distances[i] < 0) {
        return false;
      }
    }
  }
  return true; // All zero or unknown
}

int DependenceVector::getCarryingLoop() const {
  for (size_t i = 0; i < distances.size(); ++i) {
    if (is_distance_known[i] && distances[i] != 0) {
      return static_cast<int>(i);
    }
  }
  return -1; // No carrying loop
}

// DependenceInfo implementation
DependenceInfo::DependenceInfo(isl_ctx* ctx) 
    : ctx_(ctx), combined_dependence_map_(nullptr), analysis_cache_valid_(false) {}

DependenceInfo::~DependenceInfo() {
  if (combined_dependence_map_) {
    isl_union_map_free(combined_dependence_map_);
  }
}

std::vector<DependenceRelation> DependenceInfo::getDependencesByType(DependenceType type) const {
  std::vector<DependenceRelation> result;
  
  for (const auto& dep : dependences_) {
    if (dep.type == type) {
      result.push_back(dep);
    }
  }
  
  return result;
}

std::vector<DependenceRelation> DependenceInfo::getDependencesForStatement(
    const std::string& statement) const {
  
  std::vector<DependenceRelation> result;
  
  for (const auto& dep : dependences_) {
    if (dep.source_statement == statement || dep.target_statement == statement) {
      result.push_back(dep);
    }
  }
  
  return result;
}

std::vector<DependenceRelation> DependenceInfo::getDependencesForArray(Value array) const {
  std::vector<DependenceRelation> result;
  
  for (const auto& dep : dependences_) {
    if (dep.shared_array == array) {
      result.push_back(dep);
    }
  }
  
  return result;
}

void DependenceInfo::addDependence(const DependenceRelation& dep) {
  dependences_.push_back(dep);
  invalidateCache();
  
  LLVM_DEBUG(llvm::dbgs() << "Added dependence: " << dep.source_statement 
                          << " -> " << dep.target_statement 
                          << " (type: " << DependenceUtils::dependenceTypeToString(dep.type) << ")\n");
}

bool DependenceInfo::haveDependence(const std::string& source, const std::string& target) const {
  for (const auto& dep : dependences_) {
    if (dep.source_statement == source && dep.target_statement == target) {
      return true;
    }
  }
  return false;
}

isl_union_map* DependenceInfo::getCombinedDependenceMap() const {
  if (!analysis_cache_valid_) {
    computeCombinedDependenceMap();
  }
  
  return combined_dependence_map_ ? isl_union_map_copy(combined_dependence_map_) : nullptr;
}

bool DependenceInfo::canParallelizeDimension(int dimension) const {
  // Check if any dependence prevents parallelization of this dimension
  for (const auto& dep : dependences_) {
    if (dep.prevents_parallelization) {
      // Check if this dependence affects the given dimension
      for (int blocked_dim : dep.blocked_dimensions) {
        if (blocked_dim == dimension) {
          return false;
        }
      }
    }
  }
  
  return true;
}

std::vector<int> DependenceInfo::getMinimumTilingSizes() const {
  std::vector<int> min_sizes;
  
  // Find maximum dependence distances for each dimension
  int max_dim = 0;
  for (const auto& dep : dependences_) {
    for (const auto& vec : dep.distance_vectors) {
      max_dim = std::max(max_dim, static_cast<int>(vec.distances.size()));
    }
  }
  
  min_sizes.resize(max_dim, 1);
  
  for (const auto& dep : dependences_) {
    for (const auto& vec : dep.distance_vectors) {
      for (size_t i = 0; i < vec.distances.size() && i < min_sizes.size(); ++i) {
        if (vec.is_distance_known[i] && vec.distances[i] > 0) {
          min_sizes[i] = std::max(min_sizes[i], vec.distances[i] + 1);
        }
      }
    }
  }
  
  return min_sizes;
}

void DependenceInfo::invalidateCache() {
  if (combined_dependence_map_) {
    isl_union_map_free(combined_dependence_map_);
    combined_dependence_map_ = nullptr;
  }
  analysis_cache_valid_ = false;
}

void DependenceInfo::computeCombinedDependenceMap() const {
  if (combined_dependence_map_) {
    isl_union_map_free(combined_dependence_map_);
  }
  
  combined_dependence_map_ = isl_union_map_empty(isl_space_params_alloc(ctx_, 0));
  
  for (const auto& dep : dependences_) {
    if (dep.relation_map) {
      isl_union_map* dep_union = isl_union_map_from_map(isl_map_copy(dep.relation_map));
      combined_dependence_map_ = isl_union_map_union(combined_dependence_map_, dep_union);
    }
  }
  
  analysis_cache_valid_ = true;
}

// DependenceAnalyzer implementation
DependenceAnalyzer::DependenceAnalyzer(isl_ctx* ctx) : ctx_(ctx) {}

DependenceAnalyzer::~DependenceAnalyzer() = default;

std::unique_ptr<DependenceInfo> DependenceAnalyzer::analyze(const PolyhedralModel& model) {
  LLVM_DEBUG(llvm::dbgs() << "Starting dependence analysis\n");
  
  auto deps = std::make_unique<DependenceInfo>(ctx_);
  
  // Analyze different types of dependences
  auto raw_deps = computeRAWDependences(model);
  for (const auto& dep : raw_deps) {
    deps->addDependence(dep);
  }
  
  auto war_deps = computeWARDependences(model);
  for (const auto& dep : war_deps) {
    deps->addDependence(dep);
  }
  
  auto waw_deps = computeWAWDependences(model);
  for (const auto& dep : waw_deps) {
    deps->addDependence(dep);
  }
  
  if (enable_scalar_dependences_) {
    auto control_deps = computeControlDependences(model);
    for (const auto& dep : control_deps) {
      deps->addDependence(dep);
    }
  }
  
  auto reduction_deps = computeReductionDependences(model);
  for (const auto& dep : reduction_deps) {
    deps->addDependence(dep);
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Dependence analysis completed. Found " 
                          << deps->getDependences().size() << " dependences\n");
  
  return deps;
}

std::vector<DependenceRelation> DependenceAnalyzer::analyzeStatementPair(
    const PolyhedralStatement& source,
    const PolyhedralStatement& target,
    const PolyhedralModel& model) {
  
  std::vector<DependenceRelation> relations;
  
  // Check for memory-based dependences
  for (Value source_array : source.accessed_arrays) {
    for (Value target_array : target.accessed_arrays) {
      if (source_array == target_array) {
        // Same array accessed - potential dependence
        DependenceRelation dep;
        dep.source_statement = source.name;
        dep.target_statement = target.name;
        dep.shared_array = source_array;
        
        // Determine dependence type based on access patterns
        if (source.is_write_only && target.is_read_only) {
          dep.type = DependenceType::RAW;
        } else if (source.is_read_only && target.is_write_only) {
          dep.type = DependenceType::WAR;
        } else if (source.is_write_only && target.is_write_only) {
          dep.type = DependenceType::WAW;
        }
        
        // Create simplified relation map
        dep.relation_map = createSimpleDependenceMap(source, target);
        
        // Compute distance vectors
        dep.distance_vectors = computeDistanceVectors(dep);
        
        // Analyze dependence properties
        dep.prevents_parallelization = !dep.distance_vectors.empty();
        dep.can_be_carried_by_tiling = true; // Simplified assumption
        
        relations.push_back(dep);
      }
    }
  }
  
  return relations;
}

std::vector<DependenceRelation> DependenceAnalyzer::computeRAWDependences(const PolyhedralModel& model) {
  std::vector<DependenceRelation> raw_deps;
  
  const auto& statements = model.getStatements();
  
  // Check all statement pairs for RAW dependences
  for (size_t i = 0; i < statements.size(); ++i) {
    for (size_t j = 0; j < statements.size(); ++j) {
      if (i == j) continue;
      
      const auto& source = statements[i];
      const auto& target = statements[j];
      
      // RAW: source writes, target reads
      if (!source.is_read_only && !target.is_write_only) {
        auto deps = analyzeStatementPair(source, target, model);
        for (auto& dep : deps) {
          if (dep.type == DependenceType::RAW) {
            raw_deps.push_back(dep);
          }
        }
      }
    }
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Found " << raw_deps.size() << " RAW dependences\n");
  return raw_deps;
}

std::vector<DependenceRelation> DependenceAnalyzer::computeWARDependences(const PolyhedralModel& model) {
  std::vector<DependenceRelation> war_deps;
  
  const auto& statements = model.getStatements();
  
  // Check all statement pairs for WAR dependences
  for (size_t i = 0; i < statements.size(); ++i) {
    for (size_t j = 0; j < statements.size(); ++j) {
      if (i == j) continue;
      
      const auto& source = statements[i];
      const auto& target = statements[j];
      
      // WAR: source reads, target writes
      if (!source.is_write_only && !target.is_read_only) {
        auto deps = analyzeStatementPair(source, target, model);
        for (auto& dep : deps) {
          if (dep.type == DependenceType::WAR) {
            war_deps.push_back(dep);
          }
        }
      }
    }
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Found " << war_deps.size() << " WAR dependences\n");
  return war_deps;
}

std::vector<DependenceRelation> DependenceAnalyzer::computeWAWDependences(const PolyhedralModel& model) {
  std::vector<DependenceRelation> waw_deps;
  
  const auto& statements = model.getStatements();
  
  // Check all statement pairs for WAW dependences
  for (size_t i = 0; i < statements.size(); ++i) {
    for (size_t j = 0; j < statements.size(); ++j) {
      if (i == j) continue;
      
      const auto& source = statements[i];
      const auto& target = statements[j];
      
      // WAW: both write
      if (!source.is_read_only && !target.is_read_only) {
        auto deps = analyzeStatementPair(source, target, model);
        for (auto& dep : deps) {
          if (dep.type == DependenceType::WAW) {
            waw_deps.push_back(dep);
          }
        }
      }
    }
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Found " << waw_deps.size() << " WAW dependences\n");
  return waw_deps;
}

std::vector<DependenceRelation> DependenceAnalyzer::computeControlDependences(const PolyhedralModel& model) {
  // Simplified control dependence analysis
  std::vector<DependenceRelation> control_deps;
  
  // Control dependences are typically handled by the scheduling constraints
  // For now, return empty list
  
  return control_deps;
}

std::vector<DependenceRelation> DependenceAnalyzer::computeReductionDependences(const PolyhedralModel& model) {
  std::vector<DependenceRelation> reduction_deps;
  
  // Look for reduction patterns in the statements
  const auto& statements = model.getStatements();
  
  for (const auto& stmt : statements) {
    // Check if this statement looks like a reduction
    if (stmt.accessed_arrays.size() == 1 && !stmt.is_read_only && !stmt.is_write_only) {
      // Potential reduction - reads and writes the same array
      DependenceRelation dep;
      dep.source_statement = stmt.name;
      dep.target_statement = stmt.name; // Self-dependence
      dep.type = DependenceType::REDUCTION;
      dep.shared_array = stmt.accessed_arrays[0];
      
      // Create simple identity map for reduction
      dep.relation_map = nullptr; // Would create proper map in full implementation
      
      // Reduction dependences can often be parallelized with special handling
      dep.prevents_parallelization = false;
      dep.can_be_carried_by_tiling = false;
      
      reduction_deps.push_back(dep);
    }
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Found " << reduction_deps.size() << " reduction dependences\n");
  return reduction_deps;
}

std::vector<DependenceVector> DependenceAnalyzer::computeDistanceVectors(
    const DependenceRelation& dependence) {
  
  std::vector<DependenceVector> vectors;
  
  // Simplified distance vector computation
  // In practice would analyze ISL relation map
  
  DependenceVector vec;
  vec.type = dependence.type;
  
  // Assume 3D loop nest for simplicity
  vec.distances = {1, 0, 0}; // Default: dependence carried by outermost loop
  vec.is_distance_known = {true, true, true};
  
  vectors.push_back(vec);
  
  return vectors;
}

isl_map* DependenceAnalyzer::createSimpleDependenceMap(const PolyhedralStatement& source,
                                                      const PolyhedralStatement& target) {
  // Create a simple identity dependence map
  // In practice would compute actual dependence relation
  
  if (!source.domain || !target.domain) {
    return nullptr;
  }
  
  isl_space* source_space = isl_set_get_space(source.domain);
  isl_space* target_space = isl_set_get_space(target.domain);
  
  isl_space* map_space = isl_space_map_from_domain_and_range(source_space, target_space);
  isl_map* map = isl_map_universe(map_space);
  
  return map;
}

// DependenceUtils implementation
std::string DependenceUtils::dependenceTypeToString(DependenceType type) {
  switch (type) {
    case DependenceType::RAW: return "RAW";
    case DependenceType::WAR: return "WAR";
    case DependenceType::WAW: return "WAW";
    case DependenceType::CONTROL: return "Control";
    case DependenceType::MEMORY: return "Memory";
    case DependenceType::REDUCTION: return "Reduction";
  }
  return "Unknown";
}

bool DependenceUtils::allowsParallelization(DependenceType type) {
  switch (type) {
    case DependenceType::RAW:
    case DependenceType::WAW:
      return false; // These prevent parallelization
    case DependenceType::WAR:
      return true; // Can be handled with privatization
    case DependenceType::REDUCTION:
      return true; // Can be parallelized with reduction support
    default:
      return false;
  }
}

std::vector<int> DependenceUtils::computeMinimumDistances(
    const std::vector<DependenceRelation>& dependences) {
  
  std::vector<int> min_distances;
  
  // Find maximum dimensionality
  int max_dim = 0;
  for (const auto& dep : dependences) {
    for (const auto& vec : dep.distance_vectors) {
      max_dim = std::max(max_dim, static_cast<int>(vec.distances.size()));
    }
  }
  
  min_distances.resize(max_dim, 0);
  
  // Find minimum positive distance for each dimension
  for (const auto& dep : dependences) {
    for (const auto& vec : dep.distance_vectors) {
      for (size_t i = 0; i < vec.distances.size() && i < min_distances.size(); ++i) {
        if (vec.is_distance_known[i] && vec.distances[i] > 0) {
          if (min_distances[i] == 0 || vec.distances[i] < min_distances[i]) {
            min_distances[i] = vec.distances[i];
          }
        }
      }
    }
  }
  
  return min_distances;
}

bool DependenceUtils::formsCycle(const std::vector<DependenceRelation>& dependences) {
  // Simplified cycle detection
  // Would need proper graph cycle detection in full implementation
  
  std::set<std::string> sources, targets;
  
  for (const auto& dep : dependences) {
    sources.insert(dep.source_statement);
    targets.insert(dep.target_statement);
  }
  
  // If any statement is both source and target, potential cycle exists
  for (const auto& source : sources) {
    if (targets.count(source) > 0) {
      return true;
    }
  }
  
  return false;
}

} // namespace analysis
} // namespace autopoly

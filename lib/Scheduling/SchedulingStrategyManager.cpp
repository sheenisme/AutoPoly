//===- SchedulingStrategyManager.cpp - Strategy Manager -------*- C++ -*-===//
//
// Part of the AutoPoly Project
//
// This file implements the scheduling strategy manager that orchestrates
// the selection and application of appropriate scheduling strategies.
//
//===----------------------------------------------------------------------===//

#include "AutoPoly/Scheduling/SchedulingStrategy.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

#define DEBUG_TYPE "scheduling-strategy-manager"

namespace autopoly {
namespace scheduling {

SchedulingStrategyManager::SchedulingStrategyManager() {
  initializeBuiltinStrategies();
}

SchedulingStrategyManager::~SchedulingStrategyManager() = default;

void SchedulingStrategyManager::initializeBuiltinStrategies() {
  LLVM_DEBUG(llvm::dbgs() << "Initializing built-in scheduling strategies\n");
  
  // Register CPU strategy
  registerStrategy(std::make_unique<CPUSchedulingStrategy>());
  
  // Register GPU strategy
  registerStrategy(std::make_unique<GPUSchedulingStrategy>());
  
  // Register OpenCL strategy
  registerStrategy(std::make_unique<OpenCLSchedulingStrategy>());
  
  // Register FPGA strategy
  registerStrategy(std::make_unique<FPGASchedulingStrategy>());
  
  LLVM_DEBUG(llvm::dbgs() << "Registered " << strategies_.size() << " built-in strategies\n");
}

void SchedulingStrategyManager::registerStrategy(std::unique_ptr<SchedulingStrategy> strategy) {
  if (!strategy) {
    LLVM_DEBUG(llvm::dbgs() << "Warning: Attempted to register null strategy\n");
    return;
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Registering strategy: " << strategy->getName() << "\n");
  strategies_.push_back(std::move(strategy));
}

const SchedulingStrategy* SchedulingStrategyManager::selectStrategy(
    const target::TargetCharacteristics& target) const {
  
  LLVM_DEBUG(llvm::dbgs() << "Selecting strategy for target: " << target.name 
                          << " (type: " << target::TargetUtils::targetTypeToString(target.type) << ")\n");
  
  // If a strategy is forced, use it
  if (!forced_strategy_name_.empty()) {
    const auto* forced = getStrategyByName(forced_strategy_name_);
    if (forced) {
      LLVM_DEBUG(llvm::dbgs() << "Using forced strategy: " << forced_strategy_name_ << "\n");
      return forced;
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Warning: Forced strategy '" << forced_strategy_name_ 
                              << "' not found, falling back to automatic selection\n");
    }
  }
  
  const SchedulingStrategy* best_strategy = nullptr;
  double best_priority = -1.0;
  
  // Find the best suitable strategy
  for (const auto& strategy : strategies_) {
    if (strategy->isSuitableForTarget(target)) {
      double priority = strategy->calculatePriority(target);
      
      LLVM_DEBUG(llvm::dbgs() << "Strategy '" << strategy->getName() 
                              << "' is suitable with priority " << priority << "\n");
      
      if (priority > best_priority) {
        best_priority = priority;
        best_strategy = strategy.get();
      }
    }
  }
  
  if (best_strategy) {
    LLVM_DEBUG(llvm::dbgs() << "Selected strategy: " << best_strategy->getName() 
                            << " with priority " << best_priority << "\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Warning: No suitable strategy found for target\n");
  }
  
  return best_strategy;
}

std::vector<const SchedulingStrategy*> SchedulingStrategyManager::getAllStrategies() const {
  std::vector<const SchedulingStrategy*> result;
  result.reserve(strategies_.size());
  
  for (const auto& strategy : strategies_) {
    result.push_back(strategy.get());
  }
  
  return result;
}

const SchedulingStrategy* SchedulingStrategyManager::getStrategyByName(const std::string& name) const {
  for (const auto& strategy : strategies_) {
    if (strategy->getName() == name) {
      return strategy.get();
    }
  }
  
  return nullptr;
}

void SchedulingStrategyManager::forceStrategy(const std::string& name) {
  LLVM_DEBUG(llvm::dbgs() << "Forcing strategy: " << name << "\n");
  forced_strategy_name_ = name;
}

void SchedulingStrategyManager::clearForcedStrategy() {
  LLVM_DEBUG(llvm::dbgs() << "Clearing forced strategy\n");
  forced_strategy_name_.clear();
}

} // namespace scheduling
} // namespace autopoly

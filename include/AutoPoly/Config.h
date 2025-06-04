//===- Config.h - AutoPoly Configuration ------------------*- C++ -*-===//
//
// Part of the AutoPoly Project
//
// This file defines configuration constants and compilation features
// for the AutoPoly polyhedral scheduling framework.
//
//===----------------------------------------------------------------------===//

#ifndef AUTOPOLY_CONFIG_H
#define AUTOPOLY_CONFIG_H

namespace autopoly {

/// Version information for AutoPoly
constexpr int AUTOPOLY_VERSION_MAJOR = 1;
constexpr int AUTOPOLY_VERSION_MINOR = 0;
constexpr int AUTOPOLY_VERSION_PATCH = 0;

/// Maximum number of dimensions supported in polyhedral operations
constexpr int MAX_POLYHEDRAL_DIMENSIONS = 32;

/// Default timeout for scheduling computations (in seconds)
constexpr int DEFAULT_SCHEDULING_TIMEOUT = 60;

/// Feature flags for different target support
constexpr bool SUPPORT_GPU = true;
constexpr bool SUPPORT_OPENCL = true;
constexpr bool SUPPORT_CPU = true;
constexpr bool SUPPORT_FPGA = false;
constexpr bool SUPPORT_CGRA = false;
constexpr bool SUPPORT_NPU = false;
constexpr bool SUPPORT_DPU = false;
constexpr bool SUPPORT_PIM = false;

/// Feature flags for scheduling algorithms
constexpr bool SUPPORT_ISL_SCHEDULING = true;
constexpr bool SUPPORT_FEAUTRIER_SCHEDULING = true;
constexpr bool SUPPORT_PLUTO_SCHEDULING = true;

/// Debug and optimization flags
#ifndef NDEBUG
constexpr bool DEBUG_MODE = true;
#else
constexpr bool DEBUG_MODE = false;
#endif

/// Enable detailed logging for debugging
constexpr bool ENABLE_DETAILED_LOGGING = DEBUG_MODE;

/// Enable aggressive optimization transformations
constexpr bool ENABLE_AGGRESSIVE_OPTIMIZATION = true;

} // namespace autopoly

#endif // AUTOPOLY_CONFIG_H

//===- PolyhedralExtractionTest.cpp - Unit Test ---------------------------===//
//
// Copyright (c) 2025 AutoPoly Contributors
// SPDX-License-Identifier: Apache-2.0
//
// This file contains unit tests for the polyhedral model extraction framework.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "AutoPoly/Analysis/PolyhedralExtraction.h"
#include "TestUtils.h"

using namespace autopoly::analysis;
using namespace autopoly::test;

class PolyhedralExtractionTest : public AnalysisTestBase {};

TEST_F(PolyhedralExtractionTest, SimpleLoopNest) {
  isl_ctx* ctx = this->getISLContext();
  auto funcOp = this->createSimpleLoopNest(2);
  PolyhedralExtractor extractor(ctx);
  auto model = extractor.extractFromFunction(funcOp);
  ASSERT_TRUE(model != nullptr);
  EXPECT_GT(model->getStatements().size(), 0);
  PolyhedralUtils::destroyContext(ctx);
}

// Use gtest main entry
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

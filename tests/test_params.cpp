#include <gtest/gtest.h>
#include "stereo_cup_volume.hpp"

TEST(Params, DefaultsAreValid)
{
    demo::Params p;
    std::string err;
    EXPECT_TRUE(demo::validateParams(p, err)) << err;
}

TEST(Params, NumDisparitiesNotMultipleOf16)
{
    demo::Params p;
    p.numDisparities = 100;
    std::string err;
    EXPECT_FALSE(demo::validateParams(p, err));
    EXPECT_FALSE(err.empty());
}

TEST(Params, NumDisparitiesZero)
{
    demo::Params p;
    p.numDisparities = 0;
    std::string err;
    EXPECT_FALSE(demo::validateParams(p, err));
}

TEST(Params, BlockSizeEven)
{
    demo::Params p;
    p.blockSize = 10;
    std::string err;
    EXPECT_FALSE(demo::validateParams(p, err));
}

TEST(Params, MedianKernelEven)
{
    demo::Params p;
    p.medianKernel = 4;
    std::string err;
    EXPECT_FALSE(demo::validateParams(p, err));
}

TEST(Params, DepthRangeInverted)
{
    demo::Params p;
    p.minDepthMm = 500.0;
    p.maxDepthMm = 100.0;
    std::string err;
    EXPECT_FALSE(demo::validateParams(p, err));
}

TEST(Params, NegativeRoiPadding)
{
    demo::Params p;
    p.roiPadding = -5;
    std::string err;
    EXPECT_FALSE(demo::validateParams(p, err));
}

TEST(Params, ValidCustomValues)
{
    demo::Params p;
    p.numDisparities = 64;
    p.blockSize      = 7;
    p.medianKernel   = 3;
    p.minDepthMm     = 100.0;
    p.maxDepthMm     = 800.0;
    p.roiPadding     = 20;

    std::string err;
    EXPECT_TRUE(demo::validateParams(p, err)) << err;
}

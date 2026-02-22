#include <gtest/gtest.h>
#include "stereo_cup_volume.hpp"

// ── clampRoi ────────────────────────────────────────────────────────────────

TEST(RoiClamp, AlreadyInside)
{
    cv::Rect roi(10, 10, 50, 50);
    auto clamped = demo::clampRoi(roi, {640, 480});
    EXPECT_EQ(clamped, roi);
}

TEST(RoiClamp, NegativeOrigin)
{
    auto clamped = demo::clampRoi({-10, -20, 50, 60}, {640, 480});
    EXPECT_EQ(clamped.x, 0);
    EXPECT_EQ(clamped.y, 0);
    EXPECT_EQ(clamped.width,  40);
    EXPECT_EQ(clamped.height, 40);
}

TEST(RoiClamp, ExceedsBothEdges)
{
    auto clamped = demo::clampRoi({600, 450, 100, 100}, {640, 480});
    EXPECT_EQ(clamped.x, 600);
    EXPECT_EQ(clamped.y, 450);
    EXPECT_EQ(clamped.width,  40);
    EXPECT_EQ(clamped.height, 30);
}

TEST(RoiClamp, CompletelyOutside)
{
    auto clamped = demo::clampRoi({700, 500, 50, 50}, {640, 480});
    EXPECT_EQ(clamped.width,  0);
    EXPECT_EQ(clamped.height, 0);
}

TEST(RoiClamp, ZeroSizeImage)
{
    auto clamped = demo::clampRoi({0, 0, 10, 10}, {0, 0});
    EXPECT_EQ(clamped.width, 0);
    EXPECT_EQ(clamped.height, 0);
}

// ── innerRoi ────────────────────────────────────────────────────────────────

TEST(InnerRoi, ShrinksByPadding)
{
    cv::Rect outer(100, 100, 200, 200);
    auto inner = demo::innerRoi(outer, 20);
    EXPECT_EQ(inner.x, 120);
    EXPECT_EQ(inner.y, 120);
    EXPECT_EQ(inner.width,  160);
    EXPECT_EQ(inner.height, 160);
}

TEST(InnerRoi, PaddingExceedsHalfSize)
{
    cv::Rect outer(50, 50, 20, 20);
    auto inner = demo::innerRoi(outer, 15);
    EXPECT_EQ(inner.width, 0);
    EXPECT_EQ(inner.height, 0);
}

TEST(InnerRoi, ZeroPadding)
{
    cv::Rect outer(10, 10, 80, 80);
    auto inner = demo::innerRoi(outer, 0);
    EXPECT_EQ(inner, outer);
}

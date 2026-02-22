#include <gtest/gtest.h>
#include "stereo_cup_volume.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <limits>

// ── Uniform patch → median == that value ────────────────────────────────────

TEST(DepthStats, UniformPatch)
{
    cv::Mat depth(10, 10, CV_32F, cv::Scalar(300.0f));
    const double med = demo::robustPlaneDepthMm(depth, {0, 0, 10, 10});
    EXPECT_NEAR(med, 300.0, 1e-6);
}

// ── NaNs and outliers are ignored ───────────────────────────────────────────

TEST(DepthStats, RobustMedianIgnoresNaNsAndOutliers)
{
    cv::Mat depth(10, 10, CV_32F, cv::Scalar(200.0f));

    depth.at<float>(0, 0) = std::numeric_limits<float>::quiet_NaN();
    depth.at<float>(1, 1) = std::numeric_limits<float>::quiet_NaN();
    depth.at<float>(2, 2) = 9999.0f;   // above maxDepth
    depth.at<float>(3, 3) = 1.0f;      // below minDepth

    const double med = demo::robustPlaneDepthMm(depth, {0, 0, 10, 10});
    EXPECT_NEAR(med, 200.0, 1e-6);
}

// ── All NaN → returns NaN ───────────────────────────────────────────────────

TEST(DepthStats, AllNaNReturnsNaN)
{
    cv::Mat depth(5, 5, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    const double med = demo::robustPlaneDepthMm(depth, {0, 0, 5, 5});
    EXPECT_TRUE(std::isnan(med));
}

// ── All out-of-range → returns NaN ──────────────────────────────────────────

TEST(DepthStats, AllOutOfRangeReturnsNaN)
{
    cv::Mat depth(5, 5, CV_32F, cv::Scalar(10000.0f));
    const double med = demo::robustPlaneDepthMm(depth, {0, 0, 5, 5}, 50.0, 1000.0);
    EXPECT_TRUE(std::isnan(med));
}

// ── Sub-ROI is respected ────────────────────────────────────────────────────

TEST(DepthStats, SubRoiIsRespected)
{
    cv::Mat depth(20, 20, CV_32F, cv::Scalar(100.0f));

    // Paint a 5×5 block at (10,10) with a different value
    depth(cv::Rect(10, 10, 5, 5)).setTo(500.0f);

    const double medFull = demo::robustPlaneDepthMm(depth, {0, 0, 20, 20});
    const double medPatch = demo::robustPlaneDepthMm(depth, {10, 10, 5, 5});

    EXPECT_NEAR(medFull, 100.0, 1e-6);
    EXPECT_NEAR(medPatch, 500.0, 1e-6);
}

// ── Single valid pixel among NaNs ───────────────────────────────────────────

TEST(DepthStats, SingleValidPixel)
{
    cv::Mat depth(3, 3, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    depth.at<float>(1, 1) = 250.0f;

    const double med = demo::robustPlaneDepthMm(depth, {0, 0, 3, 3});
    EXPECT_NEAR(med, 250.0, 1e-6);
}

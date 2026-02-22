#include <gtest/gtest.h>
#include "stereo_cup_volume.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>

static demo::StereoCalibration makeSimpleCal(double fx)
{
    demo::StereoCalibration cal{};
    cal.P1 = cv::Mat::zeros(3, 4, CV_64F);
    cal.P1.at<double>(0, 0) = fx;
    cal.P1.at<double>(1, 1) = fx;
    cal.P1.at<double>(2, 2) = 1.0;
    return cal;
}

// ── Basic radius conversion ─────────────────────────────────────────────────

TEST(Geometry, RadiusMmFromDepthAndFx)
{
    auto cal = makeSimpleCal(800.0);
    cv::Mat depth(100, 100, CV_32F, cv::Scalar(200.0f));
    cv::Vec3f circle(50.0f, 50.0f, 40.0f);

    // radius_mm = (40 * 200) / 800 = 10 mm
    EXPECT_NEAR(demo::estimateRadiusMmFromDepth(circle, depth, cal), 10.0, 1e-6);
}

// ── Larger depth → larger mm radius for same pixel radius ───────────────────

TEST(Geometry, RadiusScalesWithDepth)
{
    auto cal = makeSimpleCal(600.0);
    cv::Mat depthA(100, 100, CV_32F, cv::Scalar(300.0f));
    cv::Mat depthB(100, 100, CV_32F, cv::Scalar(600.0f));
    cv::Vec3f circle(50.0f, 50.0f, 20.0f);

    double rA = demo::estimateRadiusMmFromDepth(circle, depthA, cal);
    double rB = demo::estimateRadiusMmFromDepth(circle, depthB, cal);
    EXPECT_NEAR(rB, 2.0 * rA, 1e-6);
}

// ── NaN depth → returns 0 ──────────────────────────────────────────────────

TEST(Geometry, NaNDepthReturnsZero)
{
    auto cal = makeSimpleCal(800.0);
    cv::Mat depth(50, 50, CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    cv::Vec3f circle(25.0f, 25.0f, 10.0f);

    EXPECT_DOUBLE_EQ(demo::estimateRadiusMmFromDepth(circle, depth, cal), 0.0);
}

// ── Zero depth → returns 0 ─────────────────────────────────────────────────

TEST(Geometry, ZeroDepthReturnsZero)
{
    auto cal = makeSimpleCal(800.0);
    cv::Mat depth(50, 50, CV_32F, cv::Scalar(0.0f));
    cv::Vec3f circle(25.0f, 25.0f, 10.0f);

    EXPECT_DOUBLE_EQ(demo::estimateRadiusMmFromDepth(circle, depth, cal), 0.0);
}

// ── Circle near image border is clamped safely ──────────────────────────────

TEST(Geometry, CircleNearBorderClampsSafely)
{
    auto cal = makeSimpleCal(500.0);
    cv::Mat depth(50, 50, CV_32F, cv::Scalar(250.0f));
    cv::Vec3f circle(200.0f, 200.0f, 15.0f);  // way outside image

    double r = demo::estimateRadiusMmFromDepth(circle, depth, cal);
    EXPECT_GT(r, 0.0);
}

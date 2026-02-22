#include <gtest/gtest.h>
#include "stereo_cup_volume.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "tests/data"
#endif

static std::string dataPath(const std::string& name)
{
    return (std::filesystem::path(TEST_DATA_DIR) / name).string();
}

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        calibPath_ = dataPath("calib.yml");
        leftPath_  = dataPath("left_01.png");
        rightPath_ = dataPath("right_01.png");

        hasData_ = std::filesystem::exists(calibPath_) &&
                   std::filesystem::exists(leftPath_)  &&
                   std::filesystem::exists(rightPath_);
    }

    std::string calibPath_, leftPath_, rightPath_;
    bool hasData_ = false;
};

TEST_F(IntegrationTest, EstimatesCupWithinRange)
{
    if (!hasData_) {
        GTEST_SKIP() << "Test data not found in " << TEST_DATA_DIR
                     << " — place left_01.png, right_01.png, calib.yml there.";
    }

    auto cal = demo::loadCalibration(calibPath_);
    cv::Mat L = cv::imread(leftPath_,  cv::IMREAD_COLOR);
    cv::Mat R = cv::imread(rightPath_, cv::IMREAD_COLOR);
    ASSERT_FALSE(L.empty());
    ASSERT_FALSE(R.empty());

    demo::Params p;
    auto m = demo::estimateCup(L, R, cal, p);
    ASSERT_TRUE(m.has_value()) << "Pipeline returned no measurement";

    EXPECT_GT(m->volumeMl, 50.0);
    EXPECT_LT(m->volumeMl, 500.0);
    EXPECT_GT(m->heightMm, 0.0);
    EXPECT_GT(m->rimRadiusMm, 0.0);
}

TEST_F(IntegrationTest, PipelineRejectsBlankImages)
{
    cv::Mat blank = cv::Mat::zeros(480, 640, CV_8UC3);

    demo::StereoCalibration cal{};
    cal.P1 = cv::Mat::eye(3, 4, CV_64F);
    cal.P1.at<double>(0, 0) = 800.0;
    cal.P1.at<double>(1, 1) = 800.0;
    cal.Q  = cv::Mat::eye(4, 4, CV_64F);
    cal.imageSize = {640, 480};

    // Create identity rectification maps
    cv::Mat mapX(480, 640, CV_32FC1), mapY(480, 640, CV_32FC1);
    for (int r = 0; r < 480; ++r)
        for (int c = 0; c < 640; ++c) {
            mapX.at<float>(r, c) = static_cast<float>(c);
            mapY.at<float>(r, c) = static_cast<float>(r);
        }
    cal.map1x = mapX.clone();
    cal.map1y = mapY.clone();
    cal.map2x = mapX.clone();
    cal.map2y = mapY.clone();

    demo::Params p;
    auto result = demo::estimateCup(blank, blank, cal, p);
    EXPECT_FALSE(result.has_value()) << "Should not detect a cup in blank images";
}

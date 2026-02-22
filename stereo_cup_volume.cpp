#include "stereo_cup_volume.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <numbers>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace demo {

// ── Calibration ─────────────────────────────────────────────────────────────

StereoCalibration loadCalibration(const std::string& path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened())
        throw std::runtime_error("Cannot open calibration file: " + path);

    StereoCalibration cal;
    fs["camera_matrix_1"]  >> cal.cameraMatrix1;
    fs["dist_coeffs_1"]    >> cal.distCoeffs1;
    fs["camera_matrix_2"]  >> cal.cameraMatrix2;
    fs["dist_coeffs_2"]    >> cal.distCoeffs2;
    fs["R"]                >> cal.R;
    fs["T"]                >> cal.T;
    fs["image_size"]       >> cal.imageSize;

    cv::stereoRectify(cal.cameraMatrix1, cal.distCoeffs1,
                      cal.cameraMatrix2, cal.distCoeffs2,
                      cal.imageSize, cal.R, cal.T,
                      cal.R1, cal.R2, cal.P1, cal.P2, cal.Q,
                      cv::CALIB_ZERO_DISPARITY, 0);

    cv::initUndistortRectifyMap(cal.cameraMatrix1, cal.distCoeffs1,
                                cal.R1, cal.P1, cal.imageSize, CV_32FC1,
                                cal.map1x, cal.map1y);
    cv::initUndistortRectifyMap(cal.cameraMatrix2, cal.distCoeffs2,
                                cal.R2, cal.P2, cal.imageSize, CV_32FC1,
                                cal.map2x, cal.map2y);
    return cal;
}

// ── Parameters ──────────────────────────────────────────────────────────────

bool validateParams(const Params& p, std::string& error)
{
    if (p.numDisparities <= 0 || p.numDisparities % 16 != 0) {
        error = "numDisparities must be > 0 and divisible by 16";
        return false;
    }
    if (p.blockSize < 1 || p.blockSize % 2 == 0) {
        error = "blockSize must be odd and >= 1";
        return false;
    }
    if (p.medianKernel < 1 || p.medianKernel % 2 == 0) {
        error = "medianKernel must be odd and >= 1";
        return false;
    }
    if (p.minDepthMm >= p.maxDepthMm) {
        error = "minDepthMm must be less than maxDepthMm";
        return false;
    }
    if (p.roiPadding < 0) {
        error = "roiPadding must be >= 0";
        return false;
    }
    return true;
}

// ── Geometry / Volume ───────────────────────────────────────────────────────

double frustumVolumeMl(double R, double r, double h)
{
    // V = (π h / 3)(R² + Rr + r²)   in mm³  →  /1000 → mL
    return (std::numbers::pi * h / 3.0) * (R * R + R * r + r * r) / 1000.0;
}

// ── ROI utilities ───────────────────────────────────────────────────────────

cv::Rect clampRoi(const cv::Rect& roi, const cv::Size& imageSize)
{
    int x1 = std::max(roi.x, 0);
    int y1 = std::max(roi.y, 0);
    int x2 = std::min(roi.x + roi.width,  imageSize.width);
    int y2 = std::min(roi.y + roi.height, imageSize.height);

    if (x2 <= x1 || y2 <= y1)
        return {0, 0, 0, 0};

    return {x1, y1, x2 - x1, y2 - y1};
}

cv::Rect innerRoi(const cv::Rect& outer, int padding)
{
    return clampRoi(
        {outer.x + padding, outer.y + padding,
         outer.width - 2 * padding, outer.height - 2 * padding},
        {outer.x + outer.width, outer.y + outer.height});
}

// ── Depth statistics ────────────────────────────────────────────────────────

double robustPlaneDepthMm(const cv::Mat& depthMap,
                          const cv::Rect& roi,
                          double minDepth,
                          double maxDepth)
{
    cv::Mat patch = depthMap(roi);
    std::vector<float> valid;
    valid.reserve(static_cast<size_t>(patch.rows * patch.cols));

    for (int r = 0; r < patch.rows; ++r) {
        const auto* row = patch.ptr<float>(r);
        for (int c = 0; c < patch.cols; ++c) {
            float v = row[c];
            if (std::isfinite(v) && v >= minDepth && v <= maxDepth)
                valid.push_back(v);
        }
    }

    if (valid.empty())
        return std::numeric_limits<double>::quiet_NaN();

    auto mid = valid.begin() + static_cast<long>(valid.size() / 2);
    std::nth_element(valid.begin(), mid, valid.end());
    return static_cast<double>(*mid);
}

// ── Pixel → mm conversion ───────────────────────────────────────────────────

double estimateRadiusMmFromDepth(const cv::Vec3f& circlePx,
                                 const cv::Mat& depthMap,
                                 const StereoCalibration& cal)
{
    const double fx = cal.P1.at<double>(0, 0);
    const int cx = std::clamp(static_cast<int>(circlePx[0]), 0, depthMap.cols - 1);
    const int cy = std::clamp(static_cast<int>(circlePx[1]), 0, depthMap.rows - 1);
    const double depth = depthMap.at<float>(cy, cx);

    if (!std::isfinite(depth) || depth <= 0.0 || fx <= 0.0)
        return 0.0;

    return (static_cast<double>(circlePx[2]) * depth) / fx;
}

// ── Pipeline ────────────────────────────────────────────────────────────────

static cv::Mat computeDepthMap(const cv::Mat& leftRect,
                               const cv::Mat& rightRect,
                               const StereoCalibration& cal,
                               const Params& p)
{
    cv::Mat grayL, grayR;
    cv::cvtColor(leftRect,  grayL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rightRect, grayR, cv::COLOR_BGR2GRAY);

    auto sgbm = cv::StereoSGBM::create(
        0, p.numDisparities, p.blockSize,
        8 * grayL.channels() * p.blockSize * p.blockSize,
        32 * grayL.channels() * p.blockSize * p.blockSize,
        1, 63, 10, 100, 32,
        cv::StereoSGBM::MODE_SGBM_3WAY);

    cv::Mat disp16;
    sgbm->compute(grayL, grayR, disp16);

    cv::Mat disp;
    disp16.convertTo(disp, CV_32F, 1.0 / 16.0);

    cv::Mat points3d;
    cv::reprojectImageTo3D(disp, points3d, cal.Q, true);

    cv::Mat depth(points3d.size(), CV_32F);
    for (int r = 0; r < points3d.rows; ++r) {
        const auto* pt = points3d.ptr<cv::Vec3f>(r);
        auto* d = depth.ptr<float>(r);
        for (int c = 0; c < points3d.cols; ++c) {
            float z = pt[c][2];
            d[c] = (z > 0 && std::isfinite(z)) ? z : std::numeric_limits<float>::quiet_NaN();
        }
    }
    return depth;
}

static std::optional<cv::Vec3f> detectRimCircle(const cv::Mat& gray,
                                                const Params& p)
{
    cv::Mat blurred;
    cv::medianBlur(gray, blurred, p.medianKernel);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT,
                     p.dp, p.minDist,
                     p.cannyThresh, p.accumThresh,
                     p.minRadius, p.maxRadius);

    if (circles.empty())
        return std::nullopt;

    return circles.front();
}

std::optional<CupMeasurement> estimateCup(const cv::Mat& left,
                                          const cv::Mat& right,
                                          const StereoCalibration& cal,
                                          const Params& params)
{
    std::string err;
    if (!validateParams(params, err))
        return std::nullopt;

    cv::Mat leftRect, rightRect;
    cv::remap(left,  leftRect,  cal.map1x, cal.map1y, cv::INTER_LINEAR);
    cv::remap(right, rightRect, cal.map2x, cal.map2y, cv::INTER_LINEAR);

    cv::Mat depth = computeDepthMap(leftRect, rightRect, cal, params);

    cv::Mat grayL;
    cv::cvtColor(leftRect, grayL, cv::COLOR_BGR2GRAY);
    auto circle = detectRimCircle(grayL, params);
    if (!circle)
        return std::nullopt;

    const float cx  = (*circle)[0];
    const float cy  = (*circle)[1];
    const float rpx = (*circle)[2];

    // Rim depth: robust median in the annular band around the rim
    cv::Rect rimRoi = clampRoi(
        cv::Rect(static_cast<int>(cx - rpx), static_cast<int>(cy - rpx),
                 static_cast<int>(2 * rpx),  static_cast<int>(2 * rpx)),
        depth.size());

    double rimDepth = robustPlaneDepthMm(depth, rimRoi,
                                         params.minDepthMm, params.maxDepthMm);
    if (!std::isfinite(rimDepth))
        return std::nullopt;

    double rimRadiusMm = estimateRadiusMmFromDepth(*circle, depth, cal);
    if (rimRadiusMm <= 0.0)
        return std::nullopt;

    // Bottom depth: small region at centre, expected to be further away
    int pad = static_cast<int>(rpx * 0.3);
    cv::Rect bottomRoi = clampRoi(
        cv::Rect(static_cast<int>(cx - pad), static_cast<int>(cy - pad),
                 2 * pad, 2 * pad),
        depth.size());

    double bottomDepth = robustPlaneDepthMm(depth, bottomRoi,
                                            params.minDepthMm, params.maxDepthMm);
    if (!std::isfinite(bottomDepth))
        return std::nullopt;

    double heightMm = std::abs(bottomDepth - rimDepth);
    double bottomRadiusMm = rimRadiusMm * params.bottomFraction;

    CupMeasurement m;
    m.rimCirclePx   = *circle;
    m.rimDepthMm    = rimDepth;
    m.rimRadiusMm   = rimRadiusMm;
    m.bottomRadiusMm = bottomRadiusMm;
    m.heightMm      = heightMm;
    m.volumeMl      = frustumVolumeMl(rimRadiusMm, bottomRadiusMm, heightMm);

    return m;
}

} // namespace demo

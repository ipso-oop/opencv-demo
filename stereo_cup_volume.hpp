#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <string>

namespace demo {

// ── Calibration ─────────────────────────────────────────────────────────────

struct StereoCalibration {
    cv::Mat cameraMatrix1, distCoeffs1;
    cv::Mat cameraMatrix2, distCoeffs2;
    cv::Mat R, T;
    cv::Mat R1, R2, P1, P2, Q;
    cv::Mat map1x, map1y, map2x, map2y;
    cv::Size imageSize;
};

StereoCalibration loadCalibration(const std::string& path);

// ── Parameters ──────────────────────────────────────────────────────────────

struct Params {
    int    numDisparities = 128;
    int    blockSize      = 15;
    int    medianKernel   = 5;
    double minDepthMm     = 50.0;
    double maxDepthMm     = 1000.0;
    int    roiPadding     = 10;
    double dp             = 1.2;
    double minDist        = 80.0;
    int    cannyThresh    = 100;
    int    accumThresh    = 40;
    int    minRadius      = 30;
    int    maxRadius      = 200;
    double bottomFraction = 0.6;
};

bool validateParams(const Params& p, std::string& error);

// ── Geometry / Volume ───────────────────────────────────────────────────────

/// Truncated-cone (frustum) volume in mL.
/// V = (π h / 3)(R² + R r + r²) / 1000
double frustumVolumeMl(double R, double r, double h);

// ── ROI utilities ───────────────────────────────────────────────────────────

/// Clamp an arbitrary cv::Rect so it lies entirely within [0, imageSize).
cv::Rect clampRoi(const cv::Rect& roi, const cv::Size& imageSize);

/// Shrink a ROI inward by `padding` pixels on each side.
cv::Rect innerRoi(const cv::Rect& outer, int padding);

// ── Depth statistics ────────────────────────────────────────────────────────

/// Robust median depth (mm) inside `roi`, ignoring NaNs and values
/// outside [minDepth, maxDepth].  Returns NaN when no valid pixels exist.
double robustPlaneDepthMm(const cv::Mat& depthMap,
                          const cv::Rect& roi,
                          double minDepth = 50.0,
                          double maxDepth = 1000.0);

// ── Pixel → mm conversion ───────────────────────────────────────────────────

/// Convert a pixel-space circle (x, y, radius_px) to a radius in mm
/// using the depth at the circle centre and the focal length from P1.
double estimateRadiusMmFromDepth(const cv::Vec3f& circlePx,
                                 const cv::Mat& depthMap,
                                 const StereoCalibration& cal);

// ── Pipeline result ─────────────────────────────────────────────────────────

struct CupMeasurement {
    double   volumeMl       = 0.0;
    double   rimRadiusMm    = 0.0;
    double   bottomRadiusMm = 0.0;
    double   heightMm       = 0.0;
    double   rimDepthMm     = 0.0;
    cv::Vec3f rimCirclePx;
};

/// Full stereo pipeline: rectify → disparity → depth → detect → measure.
std::optional<CupMeasurement> estimateCup(const cv::Mat& left,
                                          const cv::Mat& right,
                                          const StereoCalibration& cal,
                                          const Params& params);

} // namespace demo

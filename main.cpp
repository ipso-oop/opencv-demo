#include "stereo_cup_volume.hpp"
#include <iostream>
#include <cstdlib>

static void usage(const char* prog)
{
    std::cerr << "Usage: " << prog
              << " <calib.yml> <left.png> <right.png>\n";
}

int main(int argc, char* argv[])
{
    if (argc < 4) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    const std::string calibPath = argv[1];
    const std::string leftPath  = argv[2];
    const std::string rightPath = argv[3];

    demo::StereoCalibration cal;
    try {
        cal = demo::loadCalibration(calibPath);
    } catch (const std::exception& e) {
        std::cerr << "Calibration error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    cv::Mat left  = cv::imread(leftPath,  cv::IMREAD_COLOR);
    cv::Mat right = cv::imread(rightPath, cv::IMREAD_COLOR);
    if (left.empty() || right.empty()) {
        std::cerr << "Cannot read input images.\n";
        return EXIT_FAILURE;
    }

    demo::Params params;
    std::string err;
    if (!demo::validateParams(params, err)) {
        std::cerr << "Invalid parameters: " << err << "\n";
        return EXIT_FAILURE;
    }

    auto result = demo::estimateCup(left, right, cal, params);
    if (!result) {
        std::cerr << "Cup detection failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "=== Cup Measurement ===\n"
              << "  Rim radius  : " << result->rimRadiusMm    << " mm\n"
              << "  Bottom radius: " << result->bottomRadiusMm << " mm\n"
              << "  Height      : " << result->heightMm       << " mm\n"
              << "  Rim depth   : " << result->rimDepthMm     << " mm\n"
              << "  Volume      : " << result->volumeMl       << " mL\n";

    return EXIT_SUCCESS;
}

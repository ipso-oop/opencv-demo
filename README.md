# Stereo Cup Volume Estimation

A C++20 / OpenCV demo that estimates the volume of a cup from a calibrated stereo camera pair.

The project is structured as a **testable library** (`cup_lib`) with a thin CLI wrapper, demonstrating engineering practices suitable for production-grade computer-vision code.

## Project structure

```
├── CMakeLists.txt              # Build system (library + demo + tests)
├── stereo_cup_volume.hpp       # Public API (pure functions, no global state)
├── stereo_cup_volume.cpp       # Implementation
├── main.cpp                    # CLI entry point
├── .github/workflows/ci.yml   # GitHub Actions CI
├── .devcontainer/              # GitHub Codespaces config
└── tests/
    ├── test_volume.cpp         # Frustum volume formula
    ├── test_depth_stats.cpp    # Robust median depth with NaN / outlier handling
    ├── test_geometry.cpp       # Pixel→mm radius conversion
    ├── test_params.cpp         # Parameter validation
    ├── test_roi.cpp            # ROI clamping & shrinking
    └── test_integration.cpp    # End-to-end pipeline (needs test data)
```

## Prerequisites

| Dependency | Version  |
|------------|----------|
| CMake      | ≥ 3.20   |
| C++ compiler | C++20  |
| OpenCV     | ≥ 4.x    |

GoogleTest is fetched automatically via `FetchContent` (v1.14.0).

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

To skip tests:

```bash
cmake -B build -DBUILD_TESTS=OFF
```

## Run the demo

```bash
./build/demo <calib.yml> <left.png> <right.png>
```

## Run tests

```bash
cd build
ctest --output-on-failure
```

Or run directly:

```bash
./build/unit_tests          # fast, deterministic, no test data required
./build/integration_tests   # needs tests/data/{calib.yml, left_01.png, right_01.png}
```

### Test data for integration tests

Place your own stereo pair and calibration file in `tests/data/`:

```
tests/data/
  ├── calib.yml
  ├── left_01.png
  └── right_01.png
```

If no test data is present the integration tests are automatically **skipped** (not failed).

## Test philosophy

| Category | What is tested | Properties |
|----------|---------------|------------|
| **Unit tests** | Volume formula, depth statistics, pixel→mm conversion, parameter validation, ROI clamping | Fast, deterministic, CI-ready, no images needed |
| **Integration tests** | Full pipeline (rectify → disparity → detect → measure) | Threshold-based assertions, graceful skip without data |

The unit tests are designed to demonstrate **engineering maturity**: edge-case handling, NaN robustness, and clear separation of testable logic from the OpenCV pipeline.

## Algorithm overview

```mermaid
flowchart TD
    subgraph Input
        A[Load StereoCalibration\nfrom YAML] --> B[Read left & right image]
    end

    B --> C{validateParams}
    C -- invalid --> X1([return nullopt]):::fail

    subgraph Rectification
        C -- valid --> D[remap left → leftRect\nremap right → rightRect]
    end

    subgraph Stereo Matching
        D --> E[cvtColor → grayL, grayR]
        E --> F[StereoSGBM::compute\n→ disparity 16-bit]
        F --> G[convertTo float ÷ 16]
    end

    subgraph Depth Reconstruction
        G --> H[reprojectImageTo3D\n disparity, Q → points3D]
        H --> I[Extract Z → depthMap\n NaN for invalid / z ≤ 0]
    end

    subgraph Rim Detection
        D --> J[medianBlur grayL]
        J --> K[HoughCircles → circles]
        K --> L{circle found?}
        L -- no --> X2([return nullopt]):::fail
        L -- yes --> M[Pick best circle\ncx, cy, r_px]
    end

    subgraph Rim Measurement
        I --> N
        M --> N[clampRoi around rim\n→ rimRoi]
        N --> O[robustPlaneDepthMm\n median, ignore NaN/outliers\n→ rimDepth]
        O --> P{rimDepth finite?}
        P -- no --> X3([return nullopt]):::fail
        P -- yes --> Q[estimateRadiusMmFromDepth\nr_mm = r_px × depth / fx]
        Q --> R{rimRadiusMm > 0?}
        R -- no --> X4([return nullopt]):::fail
    end

    subgraph Bottom Measurement
        R -- yes --> S[clampRoi at centre\n 30% of r_px → bottomRoi]
        I --> S
        S --> T[robustPlaneDepthMm\n→ bottomDepth]
        T --> U{bottomDepth finite?}
        U -- no --> X5([return nullopt]):::fail
    end

    subgraph Volume Calculation
        U -- yes --> V["heightMm = |bottomDepth − rimDepth|"]
        V --> W[bottomRadiusMm = rimRadiusMm × fraction]
        W --> Y["frustumVolumeMl\nV = (π h / 3)(R² + Rr + r²) / 1000"]
    end

    Y --> Z([Return CupMeasurement\nvolumeM, rimRadiusMm, heightMm, ...]):::ok

    classDef fail fill:#FFCDD2,stroke:#C62828,color:#B71C1C
    classDef ok fill:#C8E6C9,stroke:#2E7D32,color:#1B5E20
```

> The full PlantUML source is in [`docs/algorithm_activity.puml`](docs/algorithm_activity.puml).

## CI & Codespaces

**GitHub Actions** runs all unit tests on every push/PR — see the badge after first push:

```markdown
![CI](https://github.com/<user>/<repo>/actions/workflows/ci.yml/badge.svg)
```

**GitHub Codespaces** — click "Code → Codespaces → New" on the repo page to get a full dev environment in the browser with OpenCV pre-installed. Then:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

## License

MIT

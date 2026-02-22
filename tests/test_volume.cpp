#include <gtest/gtest.h>
#include "stereo_cup_volume.hpp"
#include <cmath>
#include <numbers>

// ── Cylinder (R == r): V = π R² h ──────────────────────────────────────────

TEST(VolumeModel, CylinderKnownValue)
{
    constexpr double R = 40.0, r = 40.0, h = 100.0;
    const double expectedMl = (std::numbers::pi * R * R * h) / 1000.0;
    EXPECT_NEAR(demo::frustumVolumeMl(R, r, h), expectedMl, 1e-6);
}

// ── Cone (r == 0): V = (π h / 3) R² ────────────────────────────────────────

TEST(VolumeModel, ConeKnownValue)
{
    constexpr double R = 30.0, r = 0.0, h = 90.0;
    const double expectedMl = (std::numbers::pi * h * R * R / 3.0) / 1000.0;
    EXPECT_NEAR(demo::frustumVolumeMl(R, r, h), expectedMl, 1e-6);
}

// ── h == 0 → volume == 0 ───────────────────────────────────────────────────

TEST(VolumeModel, ZeroHeightIsZeroVolume)
{
    EXPECT_DOUBLE_EQ(demo::frustumVolumeMl(50.0, 30.0, 0.0), 0.0);
}

// ── R == 0 and r == 0 → volume == 0 ────────────────────────────────────────

TEST(VolumeModel, ZeroRadiiIsZeroVolume)
{
    EXPECT_DOUBLE_EQ(demo::frustumVolumeMl(0.0, 0.0, 100.0), 0.0);
}

// ── General frustum with hand-calculated reference ──────────────────────────

TEST(VolumeModel, GeneralFrustum)
{
    constexpr double R = 40.0, r = 20.0, h = 80.0;
    // V = π·80/3 · (1600+800+400) / 1000
    const double expectedMl =
        (std::numbers::pi * h / 3.0) * (R * R + R * r + r * r) / 1000.0;
    EXPECT_NEAR(demo::frustumVolumeMl(R, r, h), expectedMl, 1e-6);
}

// ── Symmetry: frustum(R, r, h) == frustum(r, R, h) ─────────────────────────

TEST(VolumeModel, SymmetricInRadii)
{
    EXPECT_DOUBLE_EQ(demo::frustumVolumeMl(30.0, 50.0, 60.0),
                     demo::frustumVolumeMl(50.0, 30.0, 60.0));
}

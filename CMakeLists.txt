# CMakeLists.txt — Build configuration for the Bioacoustic VAMP Plugin Pack.
#
# Builds two shared libraries (.so) that implement the VAMP plugin interface:
#   - birdnet-vamp.so  (BirdNET v2.4 inference)
#   - perch-vamp.so    (Perch v2 inference)
#
# Requirements:
#   - CMake >= 3.16
#   - VAMP Plugin SDK (libvamp-sdk, headers in /usr/include/vamp-sdk)
#   - C++17 compatible compiler (g++)
#
# Usage:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc)

cmake_minimum_required(VERSION 3.16)
project(BioacousticVampPack)

# ── BirdNET plugin ────────────────────────────────────────────────────────────
file(GLOB BIRDNET_SOURCES "birdnet_plugin/*.cpp")

add_library(birdnet-vamp SHARED ${BIRDNET_SOURCES})

target_include_directories(birdnet-vamp PRIVATE
    /usr/include/vamp-sdk
)

target_link_libraries(birdnet-vamp
    vamp-sdk
)

set_target_properties(birdnet-vamp PROPERTIES
    PREFIX ""           # output: birdnet-vamp.so (no "lib" prefix)
    CXX_STANDARD 17
)

# ── Perch plugin ──────────────────────────────────────────────────────────────
file(GLOB PERCH_SOURCES "perch_plugin/*.cpp")

add_library(perch-vamp SHARED ${PERCH_SOURCES})

target_include_directories(perch-vamp PRIVATE
    /usr/include/vamp-sdk
)

target_link_libraries(perch-vamp
    vamp-sdk
)

set_target_properties(perch-vamp PROPERTIES
    PREFIX ""           # output: perch-vamp.so (no "lib" prefix)
    CXX_STANDARD 17
)

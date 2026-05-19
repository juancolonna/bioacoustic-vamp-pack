# CMakeLists.txt — Build configuration for the Bioacoustic VAMP Plugin Pack.
#
# This project builds three VAMP plugin shared libraries:
#   - birdnet-vamp.so     (BirdNET inference plugin)
#   - perch-vamp.so       (Perch inference plugin)
#   - surfperch-vamp.so   (SurfPerch inference plugin)
#   - yamnet-vamp.so      (YAMNet inference plugin)
#
# The build is configured to improve binary portability on Linux x86_64
# glibc-based distributions by:
#   - linking the VAMP SDK statically, when available;
#   - linking libstdc++ statically;
#   - linking libgcc statically;
#   - keeping glibc dynamically linked, which is the recommended approach
#     for Linux shared libraries and plugins.
#
# Requirements:
#   - CMake >= 3.16
#   - A C++17-compatible compiler
#   - VAMP Plugin SDK headers
#   - Static VAMP SDK library: libvamp-sdk.a
#
# Usage:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc)
#
# Prof. Juan G. Colonna, IComp, UFAM.
#
cmake_minimum_required(VERSION 3.16)
project(BioacousticVampPack)

# Locate the static VAMP SDK library.
#
# Linking against libvamp-sdk.a avoids requiring users to have
# libvamp-sdk.so installed on their systems when loading the plugin.
find_library(VAMP_SDK_STATIC
    NAMES libvamp-sdk.a
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib
    REQUIRED
)

# add_vamp_plugin(target_name source_dir)
#
# Creates a VAMP plugin shared library from all .cpp files in source_dir.
#
# Parameters:
#   target_name — Name of the generated shared object target.
#                 The PREFIX property below removes the default "lib" prefix,
#                 so a target named "birdnet-vamp" produces "birdnet-vamp.so".
#
#   source_dir  — Directory containing the plugin C++ source files.
#
# The function applies the same include paths, link libraries, C++ standard,
# and portability-oriented linker options to all plugin targets.
function(add_vamp_plugin target_name source_dir)
    # Collect all C++ source files for this plugin.
    file(GLOB PLUGIN_SOURCES "${source_dir}/*.cpp")

    # Build the plugin as a shared library (.so).
    add_library(${target_name} SHARED ${PLUGIN_SOURCES})

    # Include the VAMP SDK headers.
    target_include_directories(${target_name} PRIVATE
        /usr/include/vamp-sdk
    )

    # Link the plugin against the static VAMP SDK library.
    target_link_libraries(${target_name}
        ${VAMP_SDK_STATIC}
    )

    # Link libstdc++ and libgcc statically to reduce runtime dependencies.
    #
    # glibc is intentionally not linked statically. Keeping glibc dynamic
    # is generally safer and more compatible for Linux plugins loaded by
    # host applications such as Audacity or Sonic Visualiser.
    target_link_options(${target_name} PRIVATE
        -static-libstdc++
        -static-libgcc
        -Wl,--as-needed
    )

    # Plugin output and C++ configuration.
    set_target_properties(${target_name} PROPERTIES
        PREFIX ""                       # Output: target_name.so, not libtarget_name.so
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
endfunction()

# Build the four plugins in the package.
add_vamp_plugin(birdnet-vamp birdnet_plugin)
add_vamp_plugin(perch-vamp perch_plugin)
add_vamp_plugin(surfperch-vamp surfperch_plugin)
add_vamp_plugin(yamnet-vamp yamnet_plugin)
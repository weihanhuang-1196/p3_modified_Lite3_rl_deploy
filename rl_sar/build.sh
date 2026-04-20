#!/bin/bash
set -e

# ========================
# Init
# ========================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "${SCRIPT_DIR}/scripts/common.sh"

# ========================
# Config
# ========================
BUILD_TYPE="Release"

# ========================
# Setup
# ========================
setup_inference_runtime() {
    print_header "[Setting up Inference Runtime]"
    bash "${SCRIPT_DIR}/scripts/download_inference_runtime.sh" || true
}

setup_mujoco() {
    print_header "[Setting up MuJoCo]"
    bash "${SCRIPT_DIR}/scripts/download_mujoco.sh" || true
}

setup_robot_descriptions() {
    print_header "[Setting up Robot Descriptions]"
    bash "${SCRIPT_DIR}/scripts/download_robot_descriptions.sh" || true
}

# ========================
# CMake Build
# ========================
run_cmake_build() {
    print_header "[CMake Build]"
    print_info "Build type: ${BUILD_TYPE}"

    cmake src/rl_sar/ -B cmake_build \
        -DUSE_CMAKE=ON \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE}

    cmake --build cmake_build -j$(nproc 2>/dev/null || echo 4)

    print_success "CMake build done"
}

# ========================
# MuJoCo Build
# ========================
run_mujoco_build() {
    print_header "[MuJoCo Build]"
    print_info "Build type: ${BUILD_TYPE}"

    cmake src/rl_sar/ -B cmake_build \
        -DUSE_CMAKE=ON \
        -DUSE_MUJOCO=ON \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE}

    cmake --build cmake_build -j$(nproc 2>/dev/null || echo 4)

    print_success "MuJoCo build done"
}

# ========================
# ROS Build（核心）
# ========================
run_ros_build() {
    local packages=("$@")
    local pkg_str=$(IFS=' '; echo "${packages[*]}")

    print_header "[ROS Build]"
    print_info "Build type: ${BUILD_TYPE}"

    # 清理不兼容产物
    if [[ "$ROS_DISTRO" != "noetic" ]]; then
        rm -rf devel .catkin_tools 2>/dev/null || true
    else
        rm -rf install log 2>/dev/null || true
    fi

    # ========================
    # ROS1 (catkin)
    # ========================
    if [[ "$ROS_DISTRO" == "noetic" ]]; then
        print_info "Using catkin"

        if [ ${#packages[@]} -eq 0 ]; then
            catkin build -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
        else
            catkin build ${pkg_str} -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
        fi

    # ========================
    # ROS2 (colcon)
    # ========================
    else
        print_info "Using colcon"

        if [ ${#packages[@]} -eq 0 ]; then
            colcon build \
                --merge-install \
                --symlink-install \
                --cmake-args -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
        else
            colcon build \
                --merge-install \
                --symlink-install \
                --packages-select ${pkg_str} \
                --cmake-args -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
        fi
    fi

    print_success "ROS build done"
}

# ========================
# Clean
# ========================
clean_workspace() {
    print_header "[Clean Workspace]"

    rm -rf build/ cmake_build/ devel/ install/ log/ logs/ .catkin_tools/

    print_success "Clean done"
}

# ========================
# Usage
# ========================
show_usage() {
    echo ""
    echo "Usage: $0 [OPTIONS] [PACKAGES]"
    echo ""
    echo "Modes:"
    echo "  -m, --cmake        CMake build"
    echo "  -mj,--mujoco       MuJoCo build"
    echo "  -c, --clean        Clean workspace"
    echo ""
    echo "Build Type:"
    echo "  -d, --debug        Debug"
    echo "  -r, --release      Release (default)"
    echo "  --reldeb           RelWithDebInfo"
    echo ""
    echo "Examples:"
    echo "  $0                    # ROS Release build"
    echo "  $0 -d                 # ROS Debug build"
    echo "  $0 pkg1 pkg2          # build specific packages"
    echo "  $0 -m -r              # CMake Release"
    echo "  $0 -mj -d             # MuJoCo Debug"
    echo ""
}

# ========================
# Main
# ========================
main() {
    local packages=()
    local clean_mode=false
    local cmake_mode=false
    local mujoco_mode=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--clean) clean_mode=true; shift ;;
            -m|--cmake) cmake_mode=true; shift ;;
            -mj|--mujoco) cmake_mode=true; mujoco_mode=true; shift ;;
            -d|--debug) BUILD_TYPE="Debug"; shift ;;
            -r|--release) BUILD_TYPE="Release"; shift ;;
            --reldeb) BUILD_TYPE="RelWithDebInfo"; shift ;;
            -h|--help) show_usage; exit 0 ;;
            *) packages+=("$1"); shift ;;
        esac
    done

    # clean
    if [ "$clean_mode" = true ]; then
        clean_workspace
        exit 0
    fi

    # mujoco
    if [ "$mujoco_mode" = true ]; then
        setup_inference_runtime
        setup_robot_descriptions
        setup_mujoco
        run_mujoco_build
        exit 0
    fi

    # cmake
    if [ "$cmake_mode" = true ]; then
        setup_inference_runtime
        run_cmake_build
        exit 0
    fi

    # ROS
    if [ -z "$ROS_DISTRO" ]; then
        print_error "ROS not sourced"
        exit 1
    fi

    setup_inference_runtime
    setup_robot_descriptions
    run_ros_build "${packages[@]}"
}

main "$@"
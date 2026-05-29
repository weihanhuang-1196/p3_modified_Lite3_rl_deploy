#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "${SCRIPT_DIR}/scripts/common.sh"

PACKAGES=("robot_msgs" "robot_joint_controller" "rl_sar")

create_symlinks() {
    print_header "[Setting up package.xml symlinks]"

    for pkg in "${PACKAGES[@]}"; do
        pkg_dir=$(find src -maxdepth 2 -type d -name "$pkg" | head -n 1)

        if [ -z "$pkg_dir" ]; then
            print_warning "Package $pkg not found in src/"
            continue
        fi

        # remove existing package.xml if symbolic
        if [ -L "$pkg_dir/package.xml" ]; then
            rm -f "$pkg_dir/package.xml"
        fi

        if [[ "$ROS_DISTRO" == "humble" || "$ROS_DISTRO" == "foxy" ]]; then
            if [ -f "$pkg_dir/package.ros2.xml" ]; then
                ln -s package.ros2.xml "$pkg_dir/package.xml"
                print_success "$pkg → linked package.xml → package.ros2.xml"
            else
                print_error "$pkg missing package.ros2.xml"
            fi
        elif [[ "$ROS_DISTRO" == "noetic" ]]; then
            if [ -f "$pkg_dir/package.ros1.xml" ]; then
                ln -s package.ros1.xml "$pkg_dir/package.xml"
                print_success "$pkg → linked package.xml → package.ros1.xml"
            fi
        else
            print_error "Unsupported ROS distro: $ROS_DISTRO"
            exit 1
        fi
    done
}

clean_workspace() {
    print_header "[Cleaning Workspace]"
    find src -name "package.xml" -type l -delete
    rm -rf build/ cmake_build/ install/ log/ logs/ devel/ .catkin_tools/
    print_success "Clean completed!"
}

run_cmake_build() {
    print_header "[Running CMake Build for RK3588]"
    print_warning "NOTE: Hardware deployment, no ROS, no ONNX, no LibTorch"
    print_separator

    cmake src/rl_sar/ -B cmake_build -DUSE_CMAKE=ON -DUSE_INFERENCE=OFF
    cmake --build cmake_build -j$(nproc 2>/dev/null || echo 4)

    print_success "RK3588 CMake build completed!"
}

run_colcon_build() {
    print_header "[Running ROS2 Colcon Build]"

    if [ -z "$ROS_DISTRO" ]; then
        print_error "ROS env not detected. Run: source /opt/ros/humble/setup.bash"
        exit 1
    fi

    create_symlinks

    print_info "Building packages: ${PACKAGES[*]}"
    colcon build --merge-install --symlink-install --packages-select "${PACKAGES[@]}"

    print_success "ROS2 build completed!"
}

run_default() {
    run_colcon_build
}

show_usage() {
    print_header "[Build System Usage]"
    echo "  ./build_rk.sh           # Build ROS2"
    echo "  ./build_rk.sh -m        # Build CMake mode (no ROS)"
    echo "  ./build_rk.sh -c        # Clean"
    echo "  ./build_rk.sh -h        # Help"
}

main() {
    local cmake_mode=false
    local clean_mode=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -m) cmake_mode=true; shift ;;
            -c) clean_mode=true; shift ;;
            -h) show_usage; exit 0 ;;
            *) shift ;;
        esac
    done

    if [ "$clean_mode" = true ]; then
        clean_workspace
        exit 0
    fi

    if [ "$cmake_mode" = true ]; then
        print_info "Skipping inference (RK3588)"
        run_cmake_build
        exit 0
    fi

    run_default
}

main "$@"

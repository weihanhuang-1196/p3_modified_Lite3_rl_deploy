cmake_minimum_required(VERSION 3.8)
project(robot_hw_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(robot_msgs REQUIRED)

add_library(robot_hw_interface SHARED
  src/robot_hw_interface.cpp
)

target_include_directories(robot_hw_interface PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

ament_target_dependencies(robot_hw_interface
  rclcpp
  hardware_interface
  pluginlib
  robot_msgs
)

install(TARGETS
  robot_hw_interface
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
    DIRECTORY include/
    DESTINATION include
)

# install(
#     DIRECTORY urdf
#     DESTINATION share/robot_hw_interface
# )

# install(
#     DIRECTORY launch
#     DESTINATION share/robot_hw_interface
# )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()




pluginlib_export_plugin_description_file(
  hardware_interface
  robot_hw_interface.xml
)

ament_export_dependencies(
  hardware_interface
)


ament_package()



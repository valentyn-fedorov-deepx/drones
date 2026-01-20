cmake_minimum_required(VERSION 3.18)
project(csrt_tracker_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Find OpenCV (must be built with tracking module)
find_package(OpenCV REQUIRED)

pybind11_add_module(csrt_tracker_ext csrt_tracker_ext.cpp)

# Link against OpenCV core/video/tracking (exact components depend on your OpenCV build)
target_link_libraries(csrt_tracker_ext PRIVATE
    ${OpenCV_LIBS}
)

target_include_directories(csrt_tracker_ext PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

# On Windows, ensure the target is built as a MODULE (pybind11_add_module already does this)

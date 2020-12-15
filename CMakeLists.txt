cmake_minimum_required(VERSION 3.17)
project(projects)
find_package(OpenCV REQUIRED)
message("opencv version " ${OpenCV_VERSION})
if (NOT OpenCV_FOUND OR OpenCV_VERSION VERSION_LESS "4.4.0")
    message("OpenCV was not found (note that OpenCV version >= 4.4.0 is required)")
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(YOLO4 main.cpp YOLO.cpp YOLO.h)
link_directories(./lib)
target_link_libraries(YOLO4
        opencv_dnn
        opencv_imgproc
        opencv_highgui
        opencv_core
        opencv_imgcodecs
        opencv_video
        opencv_videoio)
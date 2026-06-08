cmake_minimum_required(VERSION 3.27)

project(multicam_capture_gui)

find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(multicam_capture_gui
    src/multicam_capture_gui.cpp
)

target_include_directories(multicam_capture_gui PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(multicam_capture_gui PRIVATE
    pthread
    ${OpenCV_LIBS}
    Qt5::Widgets
)

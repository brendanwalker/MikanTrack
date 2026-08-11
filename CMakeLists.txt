cmake_minimum_required(VERSION 3.15)

project(MikanMediaPipe VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT MSVC)
    message(FATAL_ERROR "MikanMediaPipe is Windows/MSVC only")
endif()

add_compile_options(/MP /W3)
add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS WIN32_LEAN_AND_MEAN GLM_ENABLE_EXPERIMENTAL)

set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
set(DEPS_DIR ${ROOT_DIR}/deps)
set(THIRDPARTY_DIR ${ROOT_DIR}/thirdparty)

# ---------------------------------------------------------------- deps
# SDL2 (prebuilt)
set(SDL2_DIR ${DEPS_DIR}/SDL2-2.30.10)
set(SDL2_INCLUDE_DIR ${SDL2_DIR}/include)
set(SDL2_LIBRARIES ${SDL2_DIR}/lib/x64/SDL2.lib ${SDL2_DIR}/lib/x64/SDL2main.lib)
set(SDL2_DLL ${SDL2_DIR}/lib/x64/SDL2.dll)

# GLEW (prebuilt)
set(GLEW_DIR ${DEPS_DIR}/glew-2.2.0)
set(GLEW_INCLUDE_DIR ${GLEW_DIR}/include)
set(GLEW_LIBRARIES ${GLEW_DIR}/lib/Release/x64/glew32.lib)
set(GLEW_DLL ${GLEW_DIR}/bin/Release/x64/glew32.dll)

# OpenCV (prebuilt, opencv_world)
set(OpenCV_DIR ${DEPS_DIR}/opencv/build)
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)

# ONNX Runtime w/ DirectML (from nuget package)
set(ONNXRUNTIME_DIR ${DEPS_DIR}/onnxruntime)
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/build/native/include)
set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_DIR}/runtimes/win-x64/native/onnxruntime.lib)
set(ONNXRUNTIME_DLL ${ONNXRUNTIME_DIR}/runtimes/win-x64/native/onnxruntime.dll)
set(DIRECTML_DLL ${DEPS_DIR}/directml/bin/x64-win/DirectML.dll)

# Header-only submodules
set(GLM_INCLUDE_DIR ${THIRDPARTY_DIR}/glm)
set(NLOHMANN_JSON_INCLUDE_DIR ${THIRDPARTY_DIR}/nlohmann_json/single_include)
set(RWQUEUE_INCLUDE_DIR ${THIRDPARTY_DIR}/readerwriterqueue)

# Dear ImGui (docking branch), compiled directly into the exe
set(IMGUI_DIR ${THIRDPARTY_DIR}/imgui)
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# ---------------------------------------------------------------- sources
file(GLOB APP_SRC          ${ROOT_DIR}/src/App/*.h          ${ROOT_DIR}/src/App/*.cpp)
file(GLOB UTILITY_SRC      ${ROOT_DIR}/src/Utility/*.h      ${ROOT_DIR}/src/Utility/*.cpp)
file(GLOB MATH_SRC         ${ROOT_DIR}/src/Math/*.h         ${ROOT_DIR}/src/Math/*.cpp)
file(GLOB VIDEO_IFACE_SRC  ${ROOT_DIR}/src/Video/Interfaces/*.h)
file(GLOB VIDEO_SRC        ${ROOT_DIR}/src/Video/*.h        ${ROOT_DIR}/src/Video/*.cpp)
file(GLOB VIDEO_WMF_SRC    ${ROOT_DIR}/src/Video/WMF/*.h    ${ROOT_DIR}/src/Video/WMF/*.cpp)
file(GLOB VIDEO_RS_SRC     ${ROOT_DIR}/src/Video/RealSense/*.h ${ROOT_DIR}/src/Video/RealSense/*.cpp)
file(GLOB IMU_SRC          ${ROOT_DIR}/src/Imu/*.h          ${ROOT_DIR}/src/Imu/*.cpp)
file(GLOB IMU_JOYCON_SRC   ${ROOT_DIR}/src/Imu/Joycon/*.h   ${ROOT_DIR}/src/Imu/Joycon/*.cpp)
file(GLOB CALIBRATION_SRC  ${ROOT_DIR}/src/Calibration/*.h  ${ROOT_DIR}/src/Calibration/*.cpp)
file(GLOB VISION_SRC       ${ROOT_DIR}/src/Vision/*.h       ${ROOT_DIR}/src/Vision/*.cpp)
file(GLOB TRACKING_SRC     ${ROOT_DIR}/src/Tracking/*.h     ${ROOT_DIR}/src/Tracking/*.cpp)
file(GLOB OSC_SRC          ${ROOT_DIR}/src/Osc/*.h          ${ROOT_DIR}/src/Osc/*.cpp)
file(GLOB RENDER_SRC       ${ROOT_DIR}/src/Render/*.h       ${ROOT_DIR}/src/Render/*.cpp)
file(GLOB UI_SRC           ${ROOT_DIR}/src/UI/*.h           ${ROOT_DIR}/src/UI/*.cpp)
file(GLOB TESTS_SRC        ${ROOT_DIR}/src/Tests/*.h        ${ROOT_DIR}/src/Tests/*.cpp)

add_executable(MikanMediaPipe WIN32
    ${ROOT_DIR}/src/main.cpp
    ${APP_SRC}
    ${UTILITY_SRC}
    ${MATH_SRC}
    ${VIDEO_IFACE_SRC}
    ${VIDEO_SRC}
    ${VIDEO_WMF_SRC}
    ${VIDEO_RS_SRC}
    ${IMU_SRC}
    ${IMU_JOYCON_SRC}
    ${CALIBRATION_SRC}
    ${VISION_SRC}
    ${TRACKING_SRC}
    ${OSC_SRC}
    ${RENDER_SRC}
    ${UI_SRC}
    ${TESTS_SRC}
    ${IMGUI_SOURCES}
)

source_group(TREE ${ROOT_DIR}/src PREFIX "src" FILES
    ${APP_SRC} ${UTILITY_SRC} ${MATH_SRC} ${VIDEO_IFACE_SRC} ${VIDEO_SRC}
    ${VIDEO_WMF_SRC} ${VIDEO_RS_SRC} ${IMU_SRC} ${IMU_JOYCON_SRC}
    ${CALIBRATION_SRC} ${VISION_SRC} ${TRACKING_SRC}
    ${OSC_SRC} ${RENDER_SRC} ${UI_SRC} ${TESTS_SRC} ${ROOT_DIR}/src/main.cpp)
source_group("thirdparty\\imgui" FILES ${IMGUI_SOURCES})

target_include_directories(MikanMediaPipe PRIVATE
    ${ROOT_DIR}/src
    ${ROOT_DIR}/src/App
    ${ROOT_DIR}/src/Utility
    ${ROOT_DIR}/src/Math
    ${ROOT_DIR}/src/Video
    ${ROOT_DIR}/src/Video/Interfaces
    ${ROOT_DIR}/src/Video/WMF
    ${ROOT_DIR}/src/Video/RealSense
    ${ROOT_DIR}/deps/librealsense/include
    ${ROOT_DIR}/src/Imu
    ${ROOT_DIR}/src/Imu/Joycon
    ${ROOT_DIR}/src/Calibration
    ${ROOT_DIR}/src/Vision
    ${ROOT_DIR}/src/Tracking
    ${ROOT_DIR}/src/Osc
    ${ROOT_DIR}/src/Render
    ${ROOT_DIR}/src/UI
    ${ROOT_DIR}/src/Tests
    ${SDL2_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${GLM_INCLUDE_DIR}
    ${NLOHMANN_JSON_INCLUDE_DIR}
    ${RWQUEUE_INCLUDE_DIR}
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

target_link_libraries(MikanMediaPipe PRIVATE
    ${SDL2_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARIES}
    opengl32
    # Media Foundation (webcam capture)
    mfplat mf mfuuid mfreadwrite wmcodecdspuuid
    # HID (Joy-Con / IMU devices over Bluetooth)
    hid setupapi
    # misc win32
    winmm shlwapi ws2_32
)

# ImGui GL loader: use GLEW (already linked) instead of the built-in loader
target_compile_definitions(MikanMediaPipe PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLEW GLEW_NO_GLU)

set_property(TARGET MikanMediaPipe PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:MikanMediaPipe>")
set_property(DIRECTORY ${CMAKE_CURRENT_LIST_DIR} PROPERTY VS_STARTUP_PROJECT MikanMediaPipe)

# ---------------------------------------------------------------- post-build copies
add_custom_command(TARGET MikanMediaPipe POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SDL2_DLL} $<TARGET_FILE_DIR:MikanMediaPipe>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GLEW_DLL} $<TARGET_FILE_DIR:MikanMediaPipe>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ONNXRUNTIME_DLL} $<TARGET_FILE_DIR:MikanMediaPipe>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DIRECTML_DLL} $<TARGET_FILE_DIR:MikanMediaPipe>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OpenCV_DIR}/x64/vc16/bin/opencv_world4100$<$<CONFIG:Debug>:d>.dll
        $<TARGET_FILE_DIR:MikanMediaPipe>
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${ROOT_DIR}/models $<TARGET_FILE_DIR:MikanMediaPipe>/models
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${ROOT_DIR}/resources $<TARGET_FILE_DIR:MikanMediaPipe>/resources
)

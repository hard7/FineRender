cmake_minimum_required(VERSION 2.8)
project(FineRender)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -O3)

find_package(OpenGL REQUIRED)
find_package(Qt4 COMPONENTS QtOpenGL QtGui QtCore QtTest REQUIRED)

include(${QT_USE_FILE})
qt4_wrap_cpp(MOC_META  src/render_impl.h src/task_impl.h)
set(SOURCES ${SOURCES} src/app.cpp src/render.cpp src/task.cpp src/video.cpp)
# set(HEADERS core.h frame.h output.h render.h task.h value.h)

set(LINK_LIBRARIES ${LINK_LIBRARIES} ${QT_LIBRARIES})
set(LINK_LIBRARIES ${LINK_LIBRARIES} ${OPENGL_LIBRARIES})

find_package(FFMPEG)
if(FFMPEG_FOUND)
    add_definitions(-D FFMPEG_FOUND)
#    add_definitions("-D FFMPEG_QUIET")
    include_directories(${FFMPEG_INCLUDE_DIR})
    set(LINK_LIBRARIES ${LINK_LIBRARIES} ${FFMPEG_LIBRARIES})
endif()

add_definitions(-std=c++11)

set(LIBCORE_NAME finerender)
add_library(${LIBCORE_NAME} SHARED ${SOURCES} ${MOC_META})
target_link_libraries(${LIBCORE_NAME} ${LINK_LIBRARIES})

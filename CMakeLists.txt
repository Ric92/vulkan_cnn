cmake_minimum_required(VERSION 3.15)
project(vulkan_perceptron)

set(CMAKE_CXX_STANDARD 17)

FILE(GLOB sources src/*.cpp)

add_executable(${PROJECT_NAME} main.cpp ${sources})

target_include_directories(${PROJECT_NAME} PUBLIC includes)

find_package(Vulkan REQUIRED)

target_include_directories(${PROJECT_NAME} PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan)

find_package(PkgConfig REQUIRED)
pkg_search_module(GLM REQUIRED glm)
include_directories(${GLM_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${GLM_LIBRARY_DIRS})

find_package(glfw3 REQUIRED)
include_directories(${GLFW_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${GLFW_LIBRARIES} glfw)

include_directories(shaders)


add_executable(vulkan_test test.cpp ${sources})
target_include_directories(vulkan_test PUBLIC includes ${Vulkan_INCLUDE_DIRS} ${GLM_LIBRARY_DIRS} ${Vulkan_INCLUDE_DIRS})
target_link_libraries(vulkan_test ${GLFW_LIBRARIES} glfw Vulkan::Vulkan)
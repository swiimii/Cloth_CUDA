cmake_minimum_required(VERSION 3.15)
project(Cloth LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-g")
set(CMAKE_NVCC_FLAGS ${CMAKE_NVCC_FLAGS} "-G")
set(CMAKE_LINKER g++)

find_package(CUDA)
find_package(OpenGL)
find_package(GLUT)
find_package(Threads)

cuda_add_executable(cloth
	main.cpp
	Graphics.cpp
	Physics.cu
	Input.cpp
)
target_link_libraries(cloth
	${CUDA_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT}
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
)
target_include_directories(cloth
	PUBLIC
	${OPENGL_INCLUDE_DIRS}
	${GLUT_INCLUDE_DIRECTORIES}
)

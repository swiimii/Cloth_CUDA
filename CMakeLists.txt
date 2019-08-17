cmake_minimum_required(VERSION 3.15)
project(Physics LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-g")
set(CMAKE_NVCC_FLAGS ${CMAKE_NVCC_FLAGS} "-G")
set(CMAKE_LINKER g++)

find_package(CUDA)
find_package(OpenGL)
find_package(GLUT)
find_package(Threads)

cuda_add_executable(physics
	main.cu
)
target_link_libraries(physics
	${CUDA_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT}
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
)
target_include_directories(physics
	PUBLIC
	${OPENGL_INCLUDE_DIRS}
	${GLUT_INCLUDE_DIRECTORIES}
)

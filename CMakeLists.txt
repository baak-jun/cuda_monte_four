cmake_minimum_required(VERSION 3.24)

project(cuda_monte_four LANGUAGES CXX)

include(CheckLanguage)

option(BUILD_CUDA "Build CUDA Monte Carlo executable when a CUDA compiler is available" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(connect4_board INTERFACE)
target_include_directories(connect4_board INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(BUILD_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        add_executable(monte_carlo_cuda src/monte_carlo_cuda.cu)
        target_link_libraries(monte_carlo_cuda PRIVATE connect4_board)
        set_target_properties(monte_carlo_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES native)

        add_executable(play_monte_carlo_cuda src/play_monte_carlo_cuda.cu)
        target_link_libraries(play_monte_carlo_cuda PRIVATE connect4_board)
        set_target_properties(play_monte_carlo_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES native)

        add_executable(perfect_frontier_cuda src/perfect_frontier_cuda.cu)
        target_link_libraries(perfect_frontier_cuda PRIVATE connect4_board)
        set_target_properties(perfect_frontier_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES native)
    else()
        message(WARNING "CUDA compiler not found; monte_carlo_cuda target will not be built")
    endif()
endif()

add_executable(exhaustive src/exhaustive.cpp)
target_link_libraries(exhaustive PRIVATE connect4_board)

add_executable(solver_cpu src/solver_cpu.cpp)
target_link_libraries(solver_cpu PRIVATE connect4_board)

add_executable(perfect_solver_cpu src/perfect_solver_cpu.cpp)
target_link_libraries(perfect_solver_cpu PRIVATE connect4_board)

add_executable(perfect_db_cpu src/perfect_db_cpu.cpp)
target_link_libraries(perfect_db_cpu PRIVATE connect4_board)

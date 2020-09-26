cmake_minimum_required(VERSION 3.14)
project(nextdbscan20 LANGUAGES CXX)

set(OMP ON)
set(MPI OFF)
set(CUDA OFF)
set(DEBUG ON)
set(GTEST OFF)
set(BMS OFF)

set(NDB_FILES nextdbscan.cpp nc_tree.cpp)

if(GTEST)
    set(CMAKE_CXX_STANDARD 20)
    configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    # Add googletest directly to our build. This defines
    # the gtest and gtest_main targets.
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
            ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
            EXCLUDE_FROM_ALL)
    enable_testing()
    include_directories(${GTEST_INCLUDE_DIRS})
    add_executable(exa_tests exa_test.cpp ${NDB_FILES})
    target_link_libraries(exa_tests gtest_main)
    if (OMP)
        add_definitions(-DOMP_ON)
        find_package(OpenMP REQUIRED)
        target_link_libraries(gtest_main PUBLIC OpenMP::OpenMP_CXX)
    endif()
elseif (BMS)
    if (CUDA)
        add_definitions(-DCUDA_ON)
        set(CMAKE_CUDA_STANDARD 20)
        add_executable(exa_cu_bms exa_bms.cu)
        set_target_properties(exa_cu_bms PROPERTIES
                COMPILE_FLAGS "--expt-extended-lambda")
    else()
        set(CMAKE_CXX_STANDARD 20)
        set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -fno-exceptions")
        add_executable(exa_bms exa_bms.cpp)
        if (OMP)
            add_definitions(-DOMP_ON)
            find_package(OpenMP REQUIRED)
            target_link_libraries(exa_bms PUBLIC OpenMP::OpenMP_CXX)
        endif()
    endif()
elseif (OMP)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -fno-exceptions")
    add_definitions(-DOMP_ON)
    find_package(OpenMP REQUIRED)
    add_executable(nextdbscan20 main.cpp ${NDB_FILES})
    target_link_libraries(nextdbscan20 PUBLIC OpenMP::OpenMP_CXX)
elseif(CUDA)
    add_definitions(-DCUDA_ON)
    set(CMAKE_CUDA_STANDARD 20)
    add_executable(nextdbscan20 main.cu nextdbscan.cpp nc_tree.cu)
#    add_executable(nextdbscan20 main.cu nextdbscan.cpp)
else()
    set(CMAKE_CXX_STANDARD 20)
#    set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -fno-exceptions")
#    set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++20 -Wall -Wextra -O3 -fopenmp -fno-exceptions")
#    set(GCC_COVERAGE_COMPILE_FLAGS "-Wall -Wextra -O3 -fno-exceptions")
    add_executable(nextdbscan20 main.cpp ${NDB_FILES})
#    add_executable(nextdbscan20 main.cpp nextdbscan.cpp nc_tree.cpp magma_util.cpp)
endif()

if (MPI)
    add_definitions(-DMPI_ON)
    find_package(MPI REQUIRED)
    include_directories(${MPI_INCLUDE_PATH})
    SET(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")
    set(CMAKE_CXX_FLAGS "-Wall")
    add_executable(nextdbscan20-mpi main.cpp ${NDB_FILES})
    if(MPI_COMPILE_FLAGS)
        set_target_properties(nextdbscan20-mpi PROPERTIES
                COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    endif()
    if(MPI_LINK_FLAGS)
        set_target_properties(nextdbscan20-mpi PROPERTIES
                LINK_FLAGS "${MPI_LINK_FLAGS}")
    endif()
    target_link_libraries(nextdbscan20-mpi ${MPI_LIBRARIES})
endif()
if (DEBUG)
    add_definitions(-DDEBUG_ON)
endif()



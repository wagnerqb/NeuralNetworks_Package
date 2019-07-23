cmake_minimum_required(VERSION 2.8)

SET(PROJECT_NAME "test_NeuralNet.a")
SET(MY_MAIN "../TestNeuralNetPackage/Test_NeuralNetPackage.cpp")
SET(MY_SOURCE_DIR "../NeuralNet_src")
SET(MY_TESTSOURCE_DIR "../../UnitTestPackage/UnitTestPackagesrc")
project(${PROJECT_NAME})

# get all source files
FILE(GLOB_RECURSE sourceFiles ${MY_SOURCE_DIR}/*.cpp)
FILE(GLOB_RECURSE testsourceFiles ${MY_TESTSOURCE_DIR}/*.cpp)
SET(SOURCE
  ${sourceFiles}
  ${testsourceFiles}
  )

# get all source reader files
FILE(GLOB_RECURSE headerFiles ${MY_SOURCE_DIR}/*.h)
FILE(GLOB_RECURSE testheaderFiles ${MY_TESTSOURCE_DIR}/*.h)
SET(HEADERS
  ${headerFiles}
  ${testheaderFiles}
  )

# MACRO to get all source dirs
MACRO(HEADER_DIRECTORIES return_list)
    SET(dir_list "")
    SET(new_list ${headerFiles})
    FOREACH(file_path ${new_list})
        GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
        SET(dir_list ${dir_list} ${dir_path})
    ENDFOREACH()
    SET(new_list ${testheaderFiles})
    FOREACH(file_path ${new_list})
        GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
        SET(dir_list ${dir_list} ${dir_path})
    ENDFOREACH()
    LIST(REMOVE_DUPLICATES dir_list)
    SET(${return_list} ${dir_list})
ENDMACRO()

# include directories to search for includes
HEADER_DIRECTORIES(sourceDirList)
include_directories(${sourceDirList})

#### Find MKL Library
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/)

set(BLAS_DEFINITIONS)

if ("$ENV{MKLROOT}" STREQUAL "")
    message(FATAL_ERROR "MKLROOT is not set. Please source the Intel MKL mklvars.sh file.")
endif()

# user defined options for MKL
option(MKL_USE_parallel "Use MKL parallel" True)
option(MKL_USE_sdl "Single Dynamic Library or static/dynamic" False)
set(MKL_USE_interface "lp64" CACHE STRING "for Intel(R)64 compatible arch: ilp64/lp64 or for ia32 arch: cdecl/stdcall")

if (BLA_VENDOR MATCHES "_seq")
    set(MKL_USE_parallel OFF)
else()
    set(MKL_USE_parallel ON)
endif()

find_package(MKL REQUIRED)

if (MKL_FOUND)
    set(BLAS_INCLUDE_DIR ${MKL_INCLUDE_DIR})
    set(BLAS_LIBRARIES ${MKL_LIBRARIES})
    set(BLAS_DEFINITIONS USE_MKL)
endif()

message(STATUS "BLAS Libraries: ${BLAS_LIBRARIES}")

#### END OF Find MKL Library

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -m64")
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Ofast -ffast-math -O3 -fno-rtti")
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++14 -Wall -pg -fprofile-generate -gdwarf-3")
  SET(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -pg")
  SET(CMAKE_VERBOSE_MAKEFILE ON)
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

# adicionando o nome
add_executable(${PROJECT_NAME} ${MY_MAIN} ${SOURCE} ${HEADERS})

# Linking MKL
target_link_libraries(${PROJECT_NAME} ${LIBS})
target_link_libraries(${PROJECT_NAME} ${MKL_LIBRARIES})

# IF(UNIX)
#   target_link_libraries(${PROJECT_NAME} optimized pthread lapack blas)
#   target_link_libraries(${PROJECT_NAME} debug pthread lapack blas)
# ENDIF(UNIX)
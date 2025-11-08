cmake_minimum_required(VERSION 3.12)
project(carrot_postprocessing)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_BUILD_TYPE Release)


set(LIB_SRC_FILES
    src/postprocessing.cpp
)



#suppress a warning regarding FetchContent
cmake_policy(SET CMP0135 NEW)
include(FetchContent)



add_subdirectory(./wasm-morpho EXCLUDE_FROM_ALL)


add_library(carrot_postprocessing_core STATIC 
    ${LIB_SRC_FILES}
)
# NOTE: re-using eigen from wasm-morpho
target_include_directories(
  carrot_postprocessing_core 
  PRIVATE 
  ${eigen_SOURCE_DIR}
)
set_target_properties(carrot_postprocessing_core PROPERTIES POSITION_INDEPENDENT_CODE ON)





if(DEFINED CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES "emscripten" OR DEFINED ENV{EMSCRIPTEN})
  set(RUNNING_EMSCRIPTEN TRUE)
else()
  set(RUNNING_EMSCRIPTEN FALSE)
endif()




if(NOT RUNNING_EMSCRIPTEN)


# NOTE: re-using pybind from wasm-morpho
pybind11_add_module(
  carrot_postprocessing_ext 
  python-interface.cpp 
  src/pybind-utils.cpp
  #${LIB_SRC_FILES}
)
target_link_libraries(carrot_postprocessing_ext PRIVATE carrot_postprocessing_core)
target_include_directories(
  carrot_postprocessing_ext 
  PRIVATE 
  ${eigen_SOURCE_DIR}
)


endif()




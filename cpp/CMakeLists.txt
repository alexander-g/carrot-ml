cmake_minimum_required(VERSION 3.12)
project(carrot_postprocessing)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_BUILD_TYPE Release)


set(LIB_SRC_FILES
    src/image-utils.cpp
    src/geometry.cpp
    src/postprocessing.cpp
    src/postprocessing_cells.cpp
)



#suppress a warning regarding FetchContent
cmake_policy(SET CMP0135 NEW)
include(FetchContent)



add_subdirectory(./wasm-morpho EXCLUDE_FROM_ALL)

add_subdirectory(./wasm-big-image EXCLUDE_FROM_ALL)
set_target_properties(wasm_png PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(png_static PROPERTIES POSITION_INDEPENDENT_CODE ON)


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
  wasm-morpho/src/pybind-utils.cpp
)
target_link_libraries(carrot_postprocessing_ext PRIVATE carrot_postprocessing_core)
target_link_libraries(carrot_postprocessing_ext PRIVATE morpho_core)
target_link_libraries(carrot_postprocessing_ext PRIVATE wasm_png)
target_include_directories(
  carrot_postprocessing_ext 
  PRIVATE 
  ${eigen_SOURCE_DIR}
)


endif()  # NOT RUNNING_EMSCRIPTEN



if(RUNNING_EMSCRIPTEN)

set(JSON_URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
message("Maybe downloading ${JSON_URL}")
FetchContent_Declare(json 
    URL ${JSON_URL}
)
FetchContent_MakeAvailable(json)



add_library(carrot_postprocessing_wasm STATIC 
    wasm-interface.cpp
    src/wasm-utils.cpp
)
target_include_directories(
  carrot_postprocessing_wasm PRIVATE 
  ${eigen_SOURCE_DIR}
  ${json_SOURCE_DIR}/include
)
target_link_libraries(carrot_postprocessing_wasm PRIVATE 
  carrot_postprocessing_core
  morpho_core
  wasm_png
)
# SIMD! -flto also helps
target_compile_options(wasm_png PRIVATE -O3 -msimd128 -fexceptions -flto )
target_compile_options(morpho_core PRIVATE -O3 -msimd128 -fexceptions -flto )
target_compile_options(carrot_postprocessing_core PRIVATE -O3 -msimd128 -fexceptions)
target_compile_options(carrot_postprocessing_wasm PRIVATE -O3 -msimd128 -fexceptions)



add_custom_target(carrot_postprocessing_wasm-js ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Running emcc"
    BYPRODUCTS ${CMAKE_BINARY_DIR}/carrot_postprocessing_wasm.js
    COMMAND emcc 
      libcarrot_postprocessing_wasm.a
      libcarrot_postprocessing_core.a
      wasm-big-image/libwasm_png.a
      wasm-big-image/libpng/libpng.a
      wasm-big-image/zlib/libz.a
      wasm-morpho/libmorpho_core.a
        -sEXPORT_ES6 
        -sMODULARIZE=1 
        -sSINGLE_FILE=1 
        -sEXPORT_NAME="carrot_postprocessing_wasm" 
        -sEXPORTED_RUNTIME_METHODS=addFunction,Asyncify
        -sEXPORTED_FUNCTIONS=_postprocess_combined_wasm,_resize_mask,_free_output,_malloc,_free
        -sENVIRONMENT=web 
        -sALLOW_TABLE_GROWTH 
        -sALLOW_MEMORY_GROWTH=1
        -sWASM_BIGINT 
        -sASYNCIFY 
        -fexceptions
        -O3
        -o carrot_postprocessing_wasm.js
    DEPENDS carrot_postprocessing_wasm
)

endif()  # RUNNING_EMSCRIPTEN



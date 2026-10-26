# aphfw_cpp/utils/CMakeLists.txt

add_library(aph_utils
    timer.cpp
)

target_include_directories(aph_utils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(NODEBUGPRINT)
    target_compile_definitions(aph_utils PUBLIC NODEBUGPRINT)
endif()

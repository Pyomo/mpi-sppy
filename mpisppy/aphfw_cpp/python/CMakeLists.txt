# CMakeLists.txt for pybind11 Python module
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

if(USE_GUROBI)
    pybind11_add_module(aphfw_cpp aphfw_bindings.cpp)
    target_include_directories(aphfw_cpp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../FWAPH
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../utils
        ${CMAKE_CURRENT_SOURCE_DIR}/../config
    )
    target_link_libraries(aphfw_cpp PRIVATE aph_core_gurobi solver_gurobi)
    if(TARGET GUROBI::cxx AND TARGET GUROBI::core)
        target_link_options(aphfw_cpp PRIVATE -Wl,--no-as-needed)
        target_link_libraries(aphfw_cpp PRIVATE GUROBI::cxx GUROBI::core)
    endif()
    if(TARGET MPI::MPI_CXX)
        target_link_libraries(aphfw_cpp PRIVATE MPI::MPI_CXX)
    endif()
    target_compile_definitions(aphfw_cpp PRIVATE USE_GUROBI)
    if(NODEBUGPRINT)
        target_compile_definitions(aphfw_cpp PRIVATE NODEBUGPRINT)
    endif()
endif()

if(USE_CPLEX)
    pybind11_add_module(aphfw_cpp aphfw_bindings.cpp)
    target_include_directories(aphfw_cpp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../FWAPH
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../utils
        ${CMAKE_CURRENT_SOURCE_DIR}/../config
    )
    target_link_libraries(aphfw_cpp PRIVATE aph_core_cplex solver_cplex)
    if(TARGET MPI::MPI_CXX)
        target_link_libraries(aphfw_cpp PRIVATE MPI::MPI_CXX)
    endif()
    target_compile_definitions(aphfw_cpp PRIVATE USE_CPLEX)
    if(NODEBUGPRINT)
        target_compile_definitions(aphfw_cpp PRIVATE NODEBUGPRINT)
    endif()
endif()

if(USE_XPRESS)
    pybind11_add_module(aphfw_cpp aphfw_bindings.cpp)
    target_include_directories(aphfw_cpp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../FWAPH
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../utils
        ${CMAKE_CURRENT_SOURCE_DIR}/../config
    )
    target_link_libraries(aphfw_cpp PRIVATE aph_core_xpress solver_xpress)
    if(TARGET MPI::MPI_CXX)
        target_link_libraries(aphfw_cpp PRIVATE MPI::MPI_CXX)
    endif()
    target_compile_definitions(aphfw_cpp PRIVATE USE_XPRESS)
    if(NODEBUGPRINT)
        target_compile_definitions(aphfw_cpp PRIVATE NODEBUGPRINT)
    endif()
endif()

# pip install (scikit-build-core; see ../pyproject.toml): put the module at the
# top of site-packages, keeping the paths to the solver and MPI libraries it
# was linked against so it loads without LD_LIBRARY_PATH.
if(SKBUILD)
    set_target_properties(aphfw_cpp PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
    install(TARGETS aphfw_cpp LIBRARY DESTINATION .)
endif()

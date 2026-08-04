# Molecular-force sources are collected in a small library so every unit test does not
# have to compile the same CUDA translation units independently.  The production gpumd
# target still owns its normal source list; this library is only a test fixture.
add_library(gpumd_molecular_force_test_support STATIC
    ${PROJECT_SOURCE_DIR}/src/force/harmonic_bond.cu
    ${PROJECT_SOURCE_DIR}/src/force/molecular_force.cu
    ${PROJECT_SOURCE_DIR}/src/model/force_field_parameters.cu
    ${PROJECT_SOURCE_DIR}/src/model/harmonic_bond_data.cu
    ${PROJECT_SOURCE_DIR}/src/model/read_molecular_force.cu
    ${PROJECT_SOURCE_DIR}/src/model/topology.cu)
target_include_directories(gpumd_molecular_force_test_support PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(gpumd_molecular_force_test_support PUBLIC CUDA::cudart)

set(GPUMD_UNIT_TEST_TARGETS)

function(gpumd_add_unit_test test_name source_file)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(${test_name} PRIVATE gpumd_molecular_force_test_support)

    # These tests currently use assert(), which must remain active in Release builds.
    if(MSVC)
        target_compile_options(${test_name} PRIVATE /UNDEBUG)
    else()
        target_compile_options(${test_name} PRIVATE -UNDEBUG)
    endif()

    add_test(NAME unit.${test_name} COMMAND ${test_name})
    set_tests_properties(unit.${test_name} PROPERTIES LABELS "unit")
    list(APPEND GPUMD_UNIT_TEST_TARGETS ${test_name})
    set(GPUMD_UNIT_TEST_TARGETS "${GPUMD_UNIT_TEST_TARGETS}" PARENT_SCOPE)
endfunction()

gpumd_add_unit_test(test_topology test_topology.cpp)
gpumd_add_unit_test(test_force_field_parameters test_force_field_parameters.cpp)
gpumd_add_unit_test(test_read_molecular_force test_read_molecular_force.cpp)
gpumd_add_unit_test(test_gpu_vector test_gpu_vector.cu)
gpumd_add_unit_test(test_harmonic_bond_data test_harmonic_bond_data.cu)
gpumd_add_unit_test(test_harmonic_bond test_harmonic_bond.cu)
gpumd_add_unit_test(test_molecular_force test_molecular_force.cu)

set(GPUMD_GPU_UNIT_TESTS
    unit.test_gpu_vector
    unit.test_harmonic_bond_data
    unit.test_harmonic_bond
    unit.test_molecular_force)
set_tests_properties(${GPUMD_GPU_UNIT_TESTS} PROPERTIES
    LABELS "unit;gpu"
    RESOURCE_LOCK gpu
    SKIP_REGULAR_EXPRESSION "^SKIP: no accessible GPU")

if(UNIX)
    add_test(
        NAME integration.molecular_force_run_in
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test_molecular_force_run_in.sh
            $<TARGET_FILE:gpumd> $<TARGET_FILE:test_gpu_vector>)
    set_tests_properties(integration.molecular_force_run_in PROPERTIES
        LABELS "integration;gpu"
        RESOURCE_LOCK gpu
        SKIP_REGULAR_EXPRESSION "^SKIP: no accessible GPU"
        TIMEOUT 120)
endif()

set(GPUMD_BASELINE_EXECUTABLE "" CACHE FILEPATH
    "Optional unmodified gpumd executable used by the molecular-force no-op regression")
if(UNIX AND GPUMD_BASELINE_EXECUTABLE)
    if(EXISTS "${GPUMD_BASELINE_EXECUTABLE}")
        add_test(
            NAME regression.molecular_force_noop
            COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test_molecular_force_noop_regression.sh
                ${GPUMD_BASELINE_EXECUTABLE} $<TARGET_FILE:gpumd>)
        set_tests_properties(regression.molecular_force_noop PROPERTIES
            LABELS "regression;gpu"
            RESOURCE_LOCK gpu
            TIMEOUT 300)
    else()
        message(WARNING
            "GPUMD_BASELINE_EXECUTABLE does not exist; the no-op regression will not be registered: "
            "${GPUMD_BASELINE_EXECUTABLE}")
    endif()
endif()

set(GPUMD_CTEST_CONFIG_ARGS)
if(CMAKE_CONFIGURATION_TYPES)
    set(GPUMD_CTEST_CONFIG_ARGS --build-config $<CONFIG>)
endif()

add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure ${GPUMD_CTEST_CONFIG_ARGS}
    DEPENDS gpumd ${GPUMD_UNIT_TEST_TARGETS}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    USES_TERMINAL)

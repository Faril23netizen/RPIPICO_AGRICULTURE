# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
# ======================================================
#  Raspberry Pi Pico QDNN AIoT Project (TinyML + FreeRTOS)
# ======================================================

cmake_minimum_required(VERSION 3.13)

# Set C and C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Board configuration
set(PICO_BOARD pico CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

# Project name
project(QDNN_AIOT C CXX ASM)

# Disable default new/delete overrides
add_compile_definitions(PICO_CXX_DISABLE_ALLOCATION_OVERRIDES)

# Initialize Pico SDK
pico_sdk_init()

# ======================================================
# FreeRTOS Configuration
# ======================================================
set(FREERTOS_KERNEL_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/FreeRTOS-Kernel" CACHE STRING "FreeRTOS path")
set(FREERTOS_CONFIG_FILE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/port/FreeRTOS-Kernel" CACHE STRING "FreeRTOS config path")

include("${FREERTOS_CONFIG_FILE_DIRECTORY}/freertos_config.cmake")
include(FreeRTOS_Kernel_import.cmake)

# ======================================================
# TensorFlow Lite Micro (Official Pico Port)
# ======================================================
set(PICO_TFLMICRO_PATH "${CMAKE_CURRENT_LIST_DIR}/lib/pico-tflmicro")
add_subdirectory(${PICO_TFLMICRO_PATH})
include_directories(${PICO_TFLMICRO_PATH})

# ======================================================
# Executable Source Files
# ======================================================
add_executable(QDNN_AIOT
    src/main.cpp
    src/dht11-pico.h
    src/dht11-pico.cpp
    port/FreeRTOS-Kernel/cppMemory.cpp
    model/qdnn_fan_model.h
    model/qdnn_pump_model.h
)

# ======================================================
# Include Directories
# ======================================================
target_include_directories(QDNN_AIOT PRIVATE
    src
    model
    port/FreeRTOS-Kernel
    ${FREERTOS_CONFIG_FILE_DIRECTORY}
    ${PICO_TFLMICRO_PATH}
)

# ======================================================
# Link Libraries
# ======================================================
target_link_libraries(QDNN_AIOT
    pico_stdlib
    FreeRTOS-Kernel-Heap4
    freertos_config
    hardware_adc
    pico-tflmicro
)

# Enable USB and UART output
pico_enable_stdio_usb(QDNN_AIOT 1)
pico_enable_stdio_uart(QDNN_AIOT 1)

# ======================================================
# Extra Outputs (UF2, BIN, HEX)
# ======================================================
pico_add_extra_outputs(QDNN_AIOT)

# ======================================================
# Deployment (Optional)
# ======================================================
install(CODE "execute_process(COMMAND $ENV{HOME}/bin/picoDeploy.sh ${CMAKE_CURRENT_BINARY_DIR}/QDNN_AIOT.elf)")

# Package the UF2 file
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/QDNN_AIOT.uf2
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

# ======================================================
# Packaging
# ======================================================
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
set(CPACK_GENERATOR "ZIP" "TGZ")
include(CPack)

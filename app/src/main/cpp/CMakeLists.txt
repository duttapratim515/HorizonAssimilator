cmake_minimum_required(VERSION 3.22.1)

project(horizon_gguf_engine)

add_library(
        horizon_gguf_engine
        SHARED
        horizon_gguf_converter.cpp
        horizon_gguf_engine.cpp
        horizon_gguf_writer.cpp
)

target_compile_features(horizon_gguf_engine PRIVATE cxx_std_17)

target_link_libraries(
        horizon_gguf_engine
        android
        log
)

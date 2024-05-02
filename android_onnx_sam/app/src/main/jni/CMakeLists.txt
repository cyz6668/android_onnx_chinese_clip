project(android_onnx_sam)

cmake_minimum_required(VERSION 3.10)


add_library(tokenizerlib SHARED tokenizer.cpp helper.cpp helper.h)

target_link_libraries(tokenizerlib PRIVATE log)


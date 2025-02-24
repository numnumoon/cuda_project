# 最低限のCMakeバージョン
cmake_minimum_required(VERSION 3.18)

# プロジェクト名と使用言語
project(cuda_project LANGUAGES CXX CUDA)

# CUDAのコンパイル設定
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# CUDAインクルードパスを指定
include_directories("/usr/local/cuda/include")  # CUDAインストールパスに合わせて変更

# CUDAライブラリパスを指定
link_directories("/usr/local/cuda/lib64")  # CUDAインストールパスに合わせて変更

# 実行ファイル名とソースファイル
add_executable(cuda_project main.cpp kernel.cu)

# アーキテクチャの指定 (必要に応じて変更してください)
set_target_properties(cuda_project PROPERTIES
    CUDA_ARCHITECTURES 52
)

# CUDAランタイムのリンク
target_link_libraries(cuda_project cuda)

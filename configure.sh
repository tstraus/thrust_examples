#!/bin/bash

mkdir build
cd build

cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr ..

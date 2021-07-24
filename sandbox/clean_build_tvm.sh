#!/bin/bash
TVM_PATH=~/soo/tvm

# Re-build TVM
rm -rf $TVM_PATH/build 
mkdir $TVM_PATH/build
cd $TVM_PATH
cp cmake/config.cmake build/
cd build
cmake ..
make clean && make -j64

#!/bin/bash
TVM_PATH=~/soo/tvm

# Re-build TVM
cd $TVM_PATH/build
cmake ..
make -j64

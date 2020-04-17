set -x -e

source ci/env.sh

which g++
which nvcc
which cmake

g++ --version
nvcc --version
cmake --version

mkdir build
cd build
cmake ..
make VERBOSE=1 
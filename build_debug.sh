echo "clang-format ..."
sh clang-format_all.sh

echo "Configuring and building SensorFusion ..."
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j2
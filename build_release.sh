echo "clang-format ..."

sh clang-format_all.sh

# if [ $1 == 0 ];then
#     echo "not pre_build 3rdParty"
# else
#     echo "Configuring and building 3rdParty/lie-spline-experiments ..."

#     cd 3rdParty/lie-spline-experiments
#     sh build_basalt.sh

#     cd ../../

#     echo "Configuring and building 3rdParty/open_vins ..."

#     cd 3rdParty/open_vins
#     mkdir build
#     cd build
#     cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install
#     make -j
#     make install

#     cd ../../../
# fi

# echo "Configuring and building 3rdParty/DatasetIO ..."

# cd 3rdParty/DatasetIO
# mkdir build
# cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release
# make -j

# cd ../../../

echo "Configuring and building SensorFusion ..."
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
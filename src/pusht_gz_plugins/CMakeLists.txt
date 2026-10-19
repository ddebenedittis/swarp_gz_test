cmake_minimum_required(VERSION 3.16)
project(pusht_gz_plugins)

find_package(ament_cmake REQUIRED)
find_package(gz_sim_vendor REQUIRED)
find_package(gz-sim REQUIRED)
find_package(gz_plugin_vendor REQUIRED)
find_package(gz-plugin REQUIRED COMPONENTS register)

add_library(PlanarCoulombFriction SHARED src/PlanarCoulombFriction.cc)
target_compile_features(PlanarCoulombFriction PRIVATE cxx_std_17)
target_link_libraries(PlanarCoulombFriction gz-sim::gz-sim gz-plugin::register)

install(TARGETS PlanarCoulombFriction LIBRARY DESTINATION lib)
ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.dsv.in")

ament_package()

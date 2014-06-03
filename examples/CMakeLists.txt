include_directories(${CMAKE_SOURCE_DIR}/tests)

add_executable(solver solver.cpp)
target_link_libraries(solver ${Boost_LIBRARIES})

add_executable(block_values block_values.cpp)
target_link_libraries(block_values ${Boost_LIBRARIES})

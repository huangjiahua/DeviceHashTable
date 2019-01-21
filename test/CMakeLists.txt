
link_directories(${PROJECT_BINARY_DIR}/lib)

add_executable (HashTableTest HashTableTest.cu )

if (VS_DEBUG)
target_compile_definitions(HashTableTest PRIVATE NEED_PAUSE=1)
endif (VS_DEBUG)

set_property(TARGET HashTableTest
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(HashTableTest PRIVATE libdevht)

add_executable (InsertBenchmark ./benchmark/InsertBenchmark.cu )

if (VS_DEBUG)
target_compile_definitions(InsertBenchmark PRIVATE NEED_PAUSE=1)
endif (VS_DEBUG)

set_property(TARGET InsertBenchmark 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(InsertBenchmark PRIVATE libdevht)

set (SRC_FILES ../src/DeviceHashTable.cu ../src/HashFunc.cu ../util/util.cu ../src/DeviceHashBucket.cu)

add_executable (addr_test AddressTest.cu ${SRC_FILES})
if (VS_DEBUG)
target_compile_definitions(addr_test PRIVATE NEED_PAUSE=1)
endif (VS_DEBUG)
set_property(TARGET addr_test 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# add_executable (insert_and_find_test HashTableTest.cu ${SRC_FILES})
# if (VS_DEBUG)
# target_compile_definitions(insert_and_find_test PRIVATE NEED_PAUSE=1)
# endif (VS_DEBUG)
# set_property(TARGET insert_and_find_test
#              PROPERTY CUDA_SEPARABLE_COMPILATION ON)
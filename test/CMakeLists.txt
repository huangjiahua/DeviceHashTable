
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
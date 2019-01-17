add_executable (HashTableTest HashTableTest.cu )
if (VS_DEBUG)
target_compile_definitions(HashTableTest PRIVATE NEED_PAUSE=1)
endif (VS_DEBUG)
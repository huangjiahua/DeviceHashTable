
set(SRC_DIR_FILES ../src/DeviceHashTable.cu ../src/HashFunc.cu)
set(UTIL_DIR_FILES ../util/util.cu)

add_executable (HashTableTest HashTableTest.cu ${SRC_DIR_FILES} ${UTIL_DIR_FILES})
if (VS_DEBUG)
target_compile_definitions(HashTableTest PRIVATE NEED_PAUSE=1)
endif (VS_DEBUG)
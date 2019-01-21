set(LIB_FILES DeviceHashTable.cu DeviceHashBucket.cu HashFunc.cu ../util/util.cu)


add_library(libdevht STATIC ${LIB_FILES})

set_target_properties( libdevht
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties ( libdevht
                        PROPERTIES OUTPUT_NAME libdevht)

set_target_properties( libdevht PROPERTIES
                       ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)




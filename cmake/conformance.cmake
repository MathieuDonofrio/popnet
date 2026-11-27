if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_compile_options(
    /Zc:__cplusplus
    /Zc:enumTypes
    /Zc:inline
    /Zc:templateScope
  )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  add_compile_options(/permissive-)
endif()

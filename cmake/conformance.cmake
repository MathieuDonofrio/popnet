if (MSVC)
  add_compile_options(
    /permissive-
    /Zc:__cplusplus
    /Zc:__STDC__
    /Zc:enumTypes
    /Zc:inline
    /Zc:templateScope
  )
endif()
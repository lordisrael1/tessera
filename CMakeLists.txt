cmake_minimum_required(VERSION 3.25)
project(tessera CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CCACHE ccache)
if(CCACHE)
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
endif()

# Warnings: the bible mandates zero warnings at this level. MSVC gets its
# nearest equivalents so the Windows-native build stays honest too.
if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wshadow -Wconversion -g)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
  else()
    # znver3 is the target laptop (Ryzen 5 PRO 5650U). CI overrides this with
    # x86-64-v3 (see ci.yml) since GitHub runners are not Zen 3.
    add_compile_options(-O3 -march=native -fno-math-errno)
  endif()
endif()

option(TESSERA_SANITIZE "ASan+UBSan" OFF)
if(TESSERA_SANITIZE AND NOT MSVC)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -O1)
  add_link_options(-fsanitize=address,undefined)
endif()

option(TESSERA_TSAN "ThreadSanitizer" OFF)
if(TESSERA_TSAN AND NOT MSVC)
  add_compile_options(-fsanitize=thread -O1 -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()

add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(bench)
enable_testing()
add_subdirectory(test)

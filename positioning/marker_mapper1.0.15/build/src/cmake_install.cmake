# Install script for directory: /home/nvidia/Downloads/marker_mapper1.0.15/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmainx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmarker_mapper.so.1.0.14"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmarker_mapper.so.1.0"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES
    "/home/nvidia/Downloads/marker_mapper1.0.15/build/src/libmarker_mapper.so.1.0.14"
    "/home/nvidia/Downloads/marker_mapper1.0.15/build/src/libmarker_mapper.so.1.0"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmarker_mapper.so.1.0.14"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmarker_mapper.so.1.0"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/home/nvidia/Downloads/aruco_installed/lib:"
           NEW_RPATH "")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmainx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmarker_mapper.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmarker_mapper.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmarker_mapper.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES "/home/nvidia/Downloads/marker_mapper1.0.15/build/src/libmarker_mapper.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmarker_mapper.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmarker_mapper.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmarker_mapper.so"
         OLD_RPATH "/home/nvidia/Downloads/aruco_installed/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmarker_mapper.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmainx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/marker_mapper" TYPE FILE FILES
    "/home/nvidia/Downloads/marker_mapper1.0.15/src/debug.h"
    "/home/nvidia/Downloads/marker_mapper1.0.15/src/mapper_types.h"
    "/home/nvidia/Downloads/marker_mapper1.0.15/src/marker_mapper_exports.h"
    "/home/nvidia/Downloads/marker_mapper1.0.15/src/markermapper.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmainx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/marker_mapper/utils" TYPE FILE FILES
    "/home/nvidia/Downloads/marker_mapper1.0.15/src/utils/kdtree.h"
    "/home/nvidia/Downloads/marker_mapper1.0.15/src/utils/nanoflann.h"
    "/home/nvidia/Downloads/marker_mapper1.0.15/src/utils/se3.h"
    "/home/nvidia/Downloads/marker_mapper1.0.15/src/utils/utils3d.h"
    "/home/nvidia/Downloads/marker_mapper1.0.15/src/utils/mm_omp.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xmainx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/marker_mapper/optimizers" TYPE FILE FILES
    "/home/nvidia/Downloads/marker_mapper1.0.15/src/optimizers/ippe.h"
    "/home/nvidia/Downloads/marker_mapper1.0.15/src/optimizers/sparselevmarq.h"
    "/home/nvidia/Downloads/marker_mapper1.0.15/src/optimizers/fullsceneoptimizer.h"
    )
endif()


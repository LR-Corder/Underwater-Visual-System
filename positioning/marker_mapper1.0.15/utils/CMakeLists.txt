# CMake file for sba's demo program
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src)
LINK_LIBRARIES(${EXTRALIBNAME}${PROJECT_NAME} ${EXTRALIBNAME}aruco)

ADD_EXECUTABLE(mapper_from_video mapper_from_video.cpp)
target_link_libraries(mapper_from_video opencv_highgui ${aruco_LIBS})
ADD_EXECUTABLE(mapper_from_images mapper_from_images.cpp)
target_link_libraries(mapper_from_images opencv_highgui ${aruco_LIBS})
ADD_EXECUTABLE(mapper_viewer mapper_viewer.cpp sgl.h  sglviewer.h)
target_link_libraries(mapper_viewer opencv_highgui ${aruco_LIBS})

INSTALL(TARGETS mapper_from_images mapper_from_video   mapper_viewer     RUNTIME DESTINATION bin)

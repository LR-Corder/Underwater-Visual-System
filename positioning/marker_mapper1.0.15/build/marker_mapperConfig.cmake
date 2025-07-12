# ===================================================================================
#  marker_mapper CMake configuration file
#
#             ** File generated automatically, do not modify **
#
#  Usage from an external project:
#    In your CMakeLists.txt, add these lines:
#
#    FIND_PACKAGE(marker_mapper REQUIRED )
#    TARGET_LINK_LIBRARIES(MY_TARGET_NAME )
#
#    This file will define the following variables:
#      - marker_mapper_LIBS          : The list of libraries to links against.
#      - marker_mapper_LIB_DIR       : The directory where lib files are. Calling LINK_DIRECTORIES
#                                with this path is NOT needed.
#      - marker_mapper_VERSION       : The  version of this PROJECT_NAME build. Example: "1.2.0"
#      - marker_mapper_VERSION_MAJOR : Major version part of VERSION. Example: "1"
#      - marker_mapper_VERSION_MINOR : Minor version part of VERSION. Example: "2"
#      - marker_mapper_VERSION_PATCH : Patch version part of VERSION. Example: "0"
#
# ===================================================================================
INCLUDE_DIRECTORIES("/usr/local/include")
SET(marker_mapper_INCLUDE_DIRS "/usr/local/include")

LINK_DIRECTORIES("/usr/local/lib")
SET(marker_mapper_LIB_DIR "/usr/local/lib")

SET(marker_mapper_LIBS /home/nvidia/Downloads/aruco_installed/lib/libaruco.so;opencv_calib3d;opencv_core;opencv_dnn;opencv_features2d;opencv_flann;opencv_gapi;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_stitching;opencv_video;opencv_videoio;opencv_calib3d;opencv_core;opencv_dnn;opencv_features2d;opencv_flann;opencv_gapi;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_stitching;opencv_video;opencv_videoio;aruco marker_mapper) 

SET(marker_mapper_FOUND YES)
SET(marker_mapper_FOUND "YES")
SET(marker_mapper_VERSION        1.0.14)
SET(marker_mapper_VERSION_MAJOR  1)
SET(marker_mapper_VERSION_MINOR  0)
SET(marker_mapper_VERSION_PATCH  14)

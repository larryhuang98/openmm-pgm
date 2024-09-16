#---------------------------------------------------
# OpenMM pGM Plugin
#
# Creates OpenMMpGM plugin library.
#
# Windows:
#   OpenMMpGM.dll
#   OpenMMpGM.lib
#   OpenMMpGM_static.lib
# Unix:
#   libOpenMMpGM.so
#   libOpenMMpGM_static.a
#----------------------------------------------------

#INCLUDE(Dart)

# The source is organized into subdirectories, but we handle them all from
# this CMakeLists file rather than letting CMake visit them as SUBDIRS.
#SET(OPENMM_PGM_PLUGIN_SOURCE_SUBDIRS . openmmapi serialization)

# Collect up information about the version of the OpenMM library we're building
# and make it available to the code so it can be built into the binaries.
SET(OPENMM_pGM_PLUGIN_SOURCE_SUBDIRS . openmmapi serialization)
SET(OPENMM_pGM_LIBRARY_NAME OpenMMpGM)
SET(OPENMM_pGM_MAJOR_VERSION 1)
SET(OPENMM_pGM_MINOR_VERSION 0)
SET(OPENMM_pGM_BUILD_VERSION 0)

ADD_DEFINITIONS(-DOPENMM_pGM_LIBRARY_NAME=${OPENMM_pGM_LIBRARY_NAME}
                -DOPENMM_pGM_MAJOR_VERSION=${OPENMM_pGM_MAJOR_VERSION}
                -DOPENMM_pGM_MINOR_VERSION=${OPENMM_pGM_MINOR_VERSION}
		          -DOPENMM_pGM_BUILD_VERSION=${OPENMM_pGM_BUILD_VERSION})

SET(SHARED_pGM_TARGET ${OPENMM_pGM_LIBRARY_NAME})
SET(STATIC_pGM_TARGET ${OPENMM_pGM_LIBRARY_NAME}_static)

# These are all the places to search for header files which are
# to be part of the API.
SET(API_pGM_INCLUDE_DIRS) # start empty
FOREACH(subdir ${OPENMM_pGM_PLUGIN_SOURCE_SUBDIRS})
    # append
    SET(API_pGM_INCLUDE_DIRS ${API_pGM_INCLUDE_DIRS}
                                     ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include
                                     ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include/internal)
ENDFOREACH(subdir)

# We'll need both *relative* path names, starting with their API_INCLUDE_DIRS,
# and absolute pathnames.
SET(API_pGM_REL_INCLUDE_FILES)   # start these out empty
SET(API_pGM_ABS_INCLUDE_FILES)

FOREACH(dir ${API_pGM_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)	# returns full pathnames
    SET(API_pGM_ABS_INCLUDE_FILES ${API_pGM_ABS_INCLUDE_FILES} ${fullpaths})

    FOREACH(pathname ${fullpaths})
        GET_FILENAME_COMPONENT(filename ${pathname} NAME)
        SET(API_pGM_REL_INCLUDE_FILES ${API_pGM_REL_INCLUDE_FILES} ${dir}/${filename})
    ENDFOREACH(pathname)
ENDFOREACH(dir)

# collect up source files
SET(SOURCE_pGM_FILES) # empty
SET(SOURCE_pGM_INCLUDE_FILES)

FOREACH(subdir ${OPENMM_pGM_PLUGIN_SOURCE_SUBDIRS})
    FILE(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*/*.cpp)
    FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    SET(SOURCE_pGM_FILES         ${SOURCE_pGM_FILES}         ${src_files})   #append
    SET(SOURCE_pGM_INCLUDE_FILES ${SOURCE_pGM_INCLUDE_FILES} ${incl_files})

    ## Make sure we find these locally before looking in OpenMM/include if
    ## OpenMM was previously installed there.
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
ENDFOREACH(subdir)

# ----------------------------------------------------------------------------


INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src)

ADD_LIBRARY(${SHARED_pGM_TARGET} SHARED ${SOURCE_pGM_FILES} ${SOURCE_pGM_INCLUDE_FILES} ${API_pGM_ABS_INCLUDE_FILES})
SET_TARGET_PROPERTIES(${SHARED_pGM_TARGET} PROPERTIES LINK_FLAGS "${EXTRA_LINK_FLAGS}" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} -DOPENMM_pGM_BUILDING_SHARED_LIBRARY -DLEPTON_BUILDING_SHARED_LIBRARY")

IF(OPENMM_BUILD_STATIC_LIB)
  ADD_LIBRARY(${STATIC_pGM_TARGET} STATIC ${SOURCE_pGM_FILES} ${SOURCE_pGM_INCLUDE_FILES} ${API_pGM_ABS_INCLUDE_FILES})
  SET_TARGET_PROPERTIES(${STATIC_pGM_TARGET} PROPERTIES LINK_FLAGS "${EXTRA_LINK_FLAGS}" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} -DOPENMM_pGM_BUILDING_STATIC_LIBRARY -DOPENMM_USE_STATIC_LIBRARIES -DOPENMM_BUILDING_STATIC_LIBRARY -DLEPTON_USE_STATIC_LIBRARIES -DLEPTON_BUILDING_STATIC_LIBRARY")
ENDIF(OPENMM_BUILD_STATIC_LIB)



# ----------------------------------------------------------------------------

# On Linux need to link to libdl
IF(NOT APPLE)
    FIND_LIBRARY(DL_LIBRARY dl)
    IF(DL_LIBRARY)
      TARGET_LINK_LIBRARIES(${SHARED_pGM_TARGET} ${DL_LIBRARY})
      IF(OPENMM_BUILD_STATIC_LIB)
        TARGET_LINK_LIBRARIES(${STATIC_pGM_TARGET} ${DL_LIBRARY})
      ENDIF(OPENMM_BUILD_STATIC_LIB)
    ENDIF(DL_LIBRARY)
ENDIF()

SET( OpenMMLib OpenMM )

TARGET_LINK_LIBRARIES( ${SHARED_pGM_TARGET} ${SHARED_TARGET} )
IF(OPENMM_BUILD_STATIC_LIB)
    TARGET_LINK_LIBRARIES( ${STATIC_pGM_TARGET} ${STATIC_TARGET} )
ENDIF(OPENMM_BUILD_STATIC_LIB)

# Which hardware platforms to build

ADD_SUBDIRECTORY(platforms/reference)



INSTALL_TARGETS(/lib RUNTIME_DIRECTORY /lib ${SHARED_pGM_TARGET})
IF(OPENMM_BUILD_STATIC_LIB)
  INSTALL_TARGETS(/lib RUNTIME_DIRECTORY /lib ${STATIC_pGM_TARGET})
ENDIF(OPENMM_BUILD_STATIC_LIB)
FILE(GLOB CORE_HEADERS     include/*.h          */include/*.h)
FILE(GLOB TOP_HEADERS      include/openmm/*.h          */include/openmm/*.h)
FILE(GLOB INTERNAL_HEADERS include/openmm/internal/*.h */include/openmm/internal/*.h )
INSTALL_FILES(/include                 FILES ${CORE_HEADERS})
INSTALL_FILES(/include/openmm          FILES ${TOP_HEADERS})
INSTALL_FILES(/include/openmm/internal FILES ${INTERNAL_HEADERS})

#
# Testing
#

#ENABLE_TESTING()
#
#IF (EXECUTABLE_OUTPUT_PATH)
#  SET (TEST_PATH ${EXECUTABLE_OUTPUT_PATH})
#ELSE (EXECUTABLE_OUTPUT_PATH)
#  SET (TEST_PATH .)
#ENDIF (EXECUTABLE_OUTPUT_PATH)

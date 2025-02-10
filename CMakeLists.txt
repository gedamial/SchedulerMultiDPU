cmake_minimum_required(VERSION 3.5)

project(Scheduler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Scheduler src/main.cpp src/functions.cpp)

# define where the headers are
target_include_directories(Scheduler PUBLIC inc) 

include(GNUInstallDirs)
install(TARGETS Scheduler
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

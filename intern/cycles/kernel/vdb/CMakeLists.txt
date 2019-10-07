
set(INC
	../..
)

set(INC_SYS

)

set(SRC
	vdb_thread.cpp
	vdb_globals.cpp
)

set(HEADER_SRC
	vdb_globals.h
	vdb_intern.h
	vdb_thread.h
)

include_directories(${INC})
include_directories(SYSTEM ${INC_SYS})

add_library(cycles_kernel_vdb ${SRC} ${HEADER_SRC})


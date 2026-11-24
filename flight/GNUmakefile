# The flight app. It builds against GRTeclyn and AMReX checkouts that it does
# not modify; by default those `nacelle fetch` puts in ../external.
GRTECLYN_HOME ?= $(realpath ../external/GRTeclyn)
AMREX_HOME ?= $(realpath ../external/amrex)

include $(GRTECLYN_HOME)/Tools/GNUMake/Make.defaults

EBASE = Flight

# GRTeclyn's setup and tagging headers use the particle interpolator.
USE_PARTICLES = TRUE

include $(GRTECLYN_HOME)/Tools/GNUMake/Make.defs

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package
include $(AMREX_HOME)/Src/AmrCore/Make.package
include $(AMREX_HOME)/Src/Amr/Make.package
include $(AMREX_HOME)/Src/Particle/Make.package

GRTECLYN_SOURCE = $(GRTECLYN_HOME)/Source

src_dirs := $(GRTECLYN_SOURCE)/CCZ4 \
            $(GRTECLYN_SOURCE)/Grids \
            $(GRTECLYN_SOURCE)/GRTeclynCore \
            $(GRTECLYN_SOURCE)/IO \
            $(GRTECLYN_SOURCE)/Maths \
            $(GRTECLYN_SOURCE)/Matter \
            $(GRTECLYN_SOURCE)/ParticleInterpolator \
            $(GRTECLYN_SOURCE)/Tagging \
            $(realpath .)

CEXE_sources += $(notdir $(foreach dir, $(src_dirs), $(wildcard $(dir)/*.cpp)))
CEXE_headers += $(notdir $(foreach dir, $(src_dirs), $(wildcard $(dir)/*.hpp)))
CEXE_headers += $(notdir $(foreach dir, $(src_dirs), $(wildcard $(dir)/*.impl.hpp)))

VPATH_LOCATIONS += $(src_dirs)
INCLUDE_LOCATIONS += $(src_dirs)

# At its default level the assembler issues the fused kernel's memory reads far
# ahead of their use, and a thread then holds more values than it has
# registers. At level 1 they stay in registers.
ifeq ($(USE_CUDA),TRUE)
%/Flight.o: CXXFLAGS += --ptxas-options=-O1
endif

include $(GRTECLYN_HOME)/Tools/GNUMake/Make.rules

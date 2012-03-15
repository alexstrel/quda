###### Local configuration:

CUDA_INSTALL_PATH = /usr/local/cuda

CPU_ARCH = x86_64  # x86 or x86_64
GPU_ARCH = sm_20   # sm_10, sm_11, sm_12, sm_13, or sm_20
GPU_EMU  = no      # set to 'yes' for device emulation (CUDA 3.0 only)
OS       = linux   # linux or osx

PYTHON = python    # python 2.5 or later required for 'make gen'
DEVICE = 0         # CUDA device to use for 'make tune'

BUILD_WILSON_DIRAC = yes		# build Wilson and clover Dirac operators?
BUILD_DOMAIN_WALL_DIRAC = no		# build domain wall Dirac operators?
BUILD_STAGGERED_DIRAC = no		# build staggered Dirac operators?
BUILD_TWISTED_MASS_DIRAC = yes		# build twisted mass Dirac operators?
BUILD_NDEG_TWISTED_MASS_DIRAC = yes	# build non-degenerate twisted mass Dirac operators?
BUILD_FATLINK = no             		# build code for computing asqtad fat links?
BUILD_GAUGE_FORCE = no			# build code for (1-loop Symanzik) gauge force?
BUILD_FERMION_FORCE = no       		# build code for asqtad fermion force?

######

INC = -I$(CUDA_INSTALL_PATH)/include

ifeq ($(strip $(CPU_ARCH)), x86_64)
  ifeq ($(strip $(OS)), osx)
    LIB = -L$(CUDA_INSTALL_PATH)/lib -lcudart
    NVCCOPT = -m64
  else
    LIB = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart
  endif
else
  LIB = -L$(CUDA_INSTALL_PATH)/lib -lcudart
  COPT = -malign-double
endif

COPT += -D__CUDA_ARCH__=$(GPU_ARCH:sm_%=%0)
NVCCOPT += -D__CUDA_ARCH__=$(GPU_ARCH:sm_%=%0)

ifeq ($(strip $(GPU_EMU)), true)
  COPT += -D__DEVICE_EMULATION__
  NVCCOPT = -deviceemu
endif

ifeq ($(strip $(GPU_ARCH)), sm_20)
  NVCCOPT += -ftz=false -prec-div=true -prec-sqrt=true
endif

ifeq ($(strip $(BUILD_WILSON_DIRAC)), yes)
  NVCCOPT += -DGPU_WILSON_DIRAC
  WILSON_DIRAC_TEST=wilson_dslash_test wilson_invert_test
endif
ifeq ($(strip $(BUILD_DOMAIN_WALL_DIRAC)), yes)
  NVCCOPT += -DGPU_DOMAIN_WALL_DIRAC
  DOMAIN_WALL_DIRAC_TEST=domain_wall_dslash_test domain_wall_invert_test
endif
ifeq ($(strip $(BUILD_STAGGERED_DIRAC)), yes)
  NVCCOPT += -DGPU_STAGGERED_DIRAC
  STAGGERED_DIRAC_TEST=staggered_dslash_test staggered_invert_test
endif
ifeq ($(strip $(BUILD_TWISTED_MASS_DIRAC)), yes)
  NVCCOPT += -DGPU_TWISTED_MASS_DIRAC -DGPU_WILSON_DIRAC
  TWISTED_MASS_DIRAC_TEST=twisted_mass_dslash_test twisted_mass_invert_test
endif
ifeq ($(strip $(BUILD_NDEG_TWISTED_MASS_DIRAC)), yes)
  NVCCOPT += -DGPU_NDEG_TWISTED_MASS_DIRAC -DGPU_TWISTED_MASS_DIRAC -DGPU_WILSON_DIRAC
  TWISTED_MASS_DIRAC_TEST=twisted_mass_dslash_test twisted_mass_invert_test
endif
ifeq ($(strip $(BUILD_FATLINK)), yes)
  NVCCOPT += -DGPU_FATLINK
  FATLINK_TEST=llfat_test
endif
ifeq ($(strip $(BUILD_GAUGE_FORCE)), yes)
  NVCCOPT += -DGPU_GAUGE_FORCE
  GAUGE_FORCE_TEST=gauge_force_test
endif
ifeq ($(strip $(BUILD_FERMION_FORCE)), yes)
  NVCCOPT += -DGPU_FERMION_FORCE
  FERMION_FORCE_TEST=fermion_force_test
endif

CC = gcc
CFLAGS = -g -Wall -O3 -std=c99 $(COPT) $(INC)
CXX = g++
CXXFLAGS = -g -Wall -O3 $(COPT) $(INC)
NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc 
NVCCFLAGS = -O3 $(NVCCOPT) -arch=$(GPU_ARCH) $(INC) #--ptxas-options=-v
LDFLAGS = -fPIC $(LIB)
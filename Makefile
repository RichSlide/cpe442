# =============================================================================
# Makefile — threadSobelFilter (Raspberry Pi 5 / Cortex-A76)
#
# Usage:
#   make              — build optimised binary
#   make debug        — build with -O0 -g for GDB
#   make clean        — remove build artefacts
#   make run          — build + run with display   (pass VIDEO=your_file.mp4)
#   make bench        — build + run without display (clean PAPI numbers)
#
# Example:
#   make bench VIDEO=sample.mp4
# =============================================================================

# ---------- toolchain --------------------------------------------------------
CXX      := g++
TARGET   := sobel
SRCS     := threadSobelFilter.cpp
OBJS     := $(SRCS:.cpp=.o)

# ---------- video file (override on command line) ----------------------------
VIDEO    ?= video.mp4

# ---------- pkg-config helpers -----------------------------------------------
OPENCV_CFLAGS  := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LIBS    := $(shell pkg-config --libs   opencv4 2>/dev/null || pkg-config --libs   opencv)

# ---------- PAPI (adjust prefix if installed elsewhere) ----------------------
PAPI_PREFIX    ?= /usr
PAPI_CFLAGS    := -I$(PAPI_PREFIX)/include
PAPI_LIBS      := -L$(PAPI_PREFIX)/lib -lpapi

# ---------- compiler flags ---------------------------------------------------
#
# -march=armv8.2-a+dotprod+fp16+rcpc
#     Targets the exact ISA supported by the Cortex-A76 on Pi 5.
#     Unlocks NEON dot-product and FP16 instructions that armv8-a won't emit.
#
# -mtune=cortex-a76
#     Optimises instruction scheduling for the A76's pipeline depths.
#
# -ffast-math
#     Allows reassociation and reciprocal approximations. Safe here because
#     all our hot paths are integer NEON — this mostly helps any FP fallbacks.
#
# -fprefetch-loop-arrays
#     Inserts automatic software prefetches for array loops the compiler can
#     analyse, complementing our manual __builtin_prefetch calls.
#
# -funroll-loops
#     Unrolls small counted loops; beneficial for the scalar remainder paths.
#

OPT_FLAGS := \
    -O3                     \
    -ffast-math             \
    -funroll-loops          \
    -fomit-frame-pointer    \
    -fprefetch-loop-arrays

WARN_FLAGS := \
    -Wall -Wextra -Wshadow

CXXFLAGS := \
    -std=c++17              \
    $(OPT_FLAGS)            \
    $(WARN_FLAGS)           \
    $(OPENCV_CFLAGS)        \
    $(PAPI_CFLAGS)

LDFLAGS  := \
    $(OPENCV_LIBS)          \
    $(PAPI_LIBS)            \
    -lpthread

# ---------- debug variant ----------------------------------------------------
DEBUG_FLAGS := -O0 -g -std=c++17 $(WARN_FLAGS) $(OPENCV_CFLAGS) $(PAPI_CFLAGS)

# =============================================================================
# Targets
# =============================================================================

.PHONY: all debug clean run bench setup

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo ""
	@echo "Build complete: ./$(TARGET)"
	@echo "Run:   make run   VIDEO=$(VIDEO)"
	@echo "Bench: make bench VIDEO=$(VIDEO)"

%.o: %.cpp threadSobelFilter.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# ---------- debug build ------------------------------------------------------
debug: $(SRCS) threadSobelFilter.h
	$(CXX) $(DEBUG_FLAGS) -o $(TARGET)_debug $(SRCS) $(LDFLAGS)
	@echo "Debug build: ./$(TARGET)_debug"

# ---------- run with display -------------------------------------------------
run: all
	@echo "Setting CPU governor to performance..."
	@echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
	sudo chrt -f 99 ./$(TARGET) $(VIDEO)

# ---------- run without display (clean benchmark numbers) --------------------
bench: all
	@echo "Setting CPU governor to performance..."
	@echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
	sudo chrt -f 99 ./$(TARGET) $(VIDEO) --no-display

# ---------- one-time Pi 5 setup (run once after fresh OS install) ------------
# Installs build dependencies, sets up PAPI perf event permissions.
setup:
	sudo apt-get update
	sudo apt-get install -y \
	    libopencv-dev       \
	    libpapi-dev         \
	    pkg-config          \
	    g++
	@echo ""
	@echo "Allowing unprivileged perf events (needed for PAPI)..."
	@echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
	@echo "To make that permanent, add to /etc/sysctl.conf:"
	@echo "    kernel.perf_event_paranoid = 1"

# ---------- clean ------------------------------------------------------------
clean:
	rm -f $(OBJS) $(TARGET) $(TARGET)_debug

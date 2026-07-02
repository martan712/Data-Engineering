# Makefile — build fused_panel_maxsim.so (Stage 3b fused panel MaxSim kernels).
#
# Detects clang++ first; falls back to g++.  Produces the .so in this directory
# so that src/bondmaxsim/kernels/fused_panel.py can locate it at:
#   cpp/fused_panel_maxsim/fused_panel_maxsim.so
#
# -fopenmp: threading over vectorgroups (kernel also builds/runs without it).
# -march=native: enables the AVX-512 fast path on this machine; a portable
#   autovectorized fallback compiles otherwise.

CXX := $(shell which clang++ 2>/dev/null || which g++ 2>/dev/null)
ifeq ($(CXX),)
$(error "No C++ compiler found (tried clang++ and g++). Install clang++ or g++.")
endif

TARGET  := fused_panel_maxsim.so
SOURCES := fused_panel_maxsim.cpp
CXXFLAGS := -O3 -march=native -DNDEBUG -std=c++20 -shared -fPIC -fopenmp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES)
	@echo "[fused_panel_maxsim] Compiler: $(CXX)"
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "[fused_panel_maxsim] Built: $(TARGET)"

clean:
	rm -f $(TARGET)

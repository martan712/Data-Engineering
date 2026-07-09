# Makefile — build fused_panel_maxsim.so (Stage 3b fused panel MaxSim kernels).
#
# Detects clang++ first; falls back to g++.  Produces the .so in this directory
# so that src/bondmaxsim/kernels/fused_panel.py can locate it at:
#   cpp/fused_panel_maxsim/fused_panel_maxsim.so
#
# Sources (one exported kernel family per TU; shared code in headers):
#   brute.cpp      — fused_panel_maxsim_brute
#   bond_doc.cpp   — fused_panel_maxsim_bond / _bond_cheap
#   bond_token.cpp — fused_panel_maxsim_bond_token
#
# -fopenmp: threading over vectorgroups (kernels also build/run without it).
# -march=native: enables the AVX-512 fast path on this machine; a portable
#   autovectorized fallback compiles otherwise.

CXX := $(shell which clang++ 2>/dev/null || which g++ 2>/dev/null)
ifeq ($(CXX),)
$(error "No C++ compiler found (tried clang++ and g++). Install clang++ or g++.")
endif

TARGET   := fused_panel_maxsim.so
SOURCES  := brute.cpp bond_doc.cpp bond_token.cpp
HEADERS  := common.hpp microkernel.hpp microkernel_avx512.hpp microkernel_portable.hpp
OBJECTS  := $(SOURCES:.cpp=.o)
CXXFLAGS := -O3 -march=native -DNDEBUG -std=c++20 -fPIC -fopenmp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@echo "[fused_panel_maxsim] Compiler: $(CXX)"
	$(CXX) -shared -fopenmp -o $@ $(OBJECTS)
	@echo "[fused_panel_maxsim] Built: $(TARGET)"

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(TARGET) $(OBJECTS)

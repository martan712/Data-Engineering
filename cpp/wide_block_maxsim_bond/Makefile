# Makefile — build wide_block_maxsim_bond.so (accounting + throughput kernels).
#
# Detects clang++ first; falls back to g++.  Produces the .so in this directory
# so that src/bondmaxsim/kernels/bindings.py can locate it at:
#   cpp/wide_block_maxsim_bond/wide_block_maxsim_bond.so
#
# Self-contained: no PDX include is needed at compile time (Stage 2 spec).

CXX := $(shell which clang++ 2>/dev/null || which g++ 2>/dev/null)
ifeq ($(CXX),)
$(error "No C++ compiler found (tried clang++ and g++). Install clang++ or g++.")
endif

TARGET  := wide_block_maxsim_bond.so
SOURCES := wide_block_maxsim_bond.cpp
CXXFLAGS := -O3 -march=native -DNDEBUG -std=c++20 -shared -fPIC

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES)
	@echo "[wide_block_maxsim_bond] Compiler: $(CXX)"
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "[wide_block_maxsim_bond] Built: $(TARGET)"

clean:
	rm -f $(TARGET)

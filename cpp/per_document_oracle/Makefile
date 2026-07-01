# Makefile — build per_document_oracle.so (accounting + throughput + brute-force kernels).
#
# Detects clang++ first; falls back to g++.  Produces the .so in this directory
# so that src/bondmaxsim/kernels/bindings.py can locate it at:
#   cpp/per_document_oracle/per_document_oracle.so

CXX := $(shell which clang++ 2>/dev/null || which g++ 2>/dev/null)
ifeq ($(CXX),)
$(error "No C++ compiler found (tried clang++ and g++). Install clang++ or g++.")
endif

TARGET  := per_document_oracle.so
SOURCES := per_document_oracle.cpp
CXXFLAGS := -O3 -march=native -DNDEBUG -std=c++20 -shared -fPIC

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES)
	@echo "[per_document_oracle] Compiler: $(CXX)"
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "[per_document_oracle] Built: $(TARGET)"

clean:
	rm -f $(TARGET)

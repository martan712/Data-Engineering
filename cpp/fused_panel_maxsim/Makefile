ifeq ($(origin CXX),default)
CXX := $(shell command -v clang++ 2>/dev/null || command -v g++ 2>/dev/null)
endif
BUILD ?= portable

TARGET := fused_panel_maxsim.so
SOURCES := brute.cpp bond_doc.cpp bond_token.cpp
HEADERS := common.hpp microkernel.hpp microkernel_avx512.hpp microkernel_portable.hpp ../native_validation.hpp
PROFILE_DIR := build/$(BUILD)
PROFILE_TARGET := $(PROFILE_DIR)/$(TARGET)
OBJECTS := $(addprefix $(PROFILE_DIR)/,$(SOURCES:.cpp=.o))
MANIFEST := $(PROFILE_DIR)/manifest.json
COMMON_FLAGS := -std=c++20 -fPIC -fopenmp

ifeq ($(CXX),)
$(error No C++ compiler found; set CXX or install clang++/g++)
endif
ifeq ($(BUILD),portable)
PROFILE_FLAGS := -O3 -march=x86-64-v2 -DNDEBUG
else ifeq ($(BUILD),paper-native)
PROFILE_FLAGS := -O3 -march=native -DNDEBUG
else ifeq ($(BUILD),sanitize)
PROFILE_FLAGS := -O1 -g -march=x86-64-v2 -fsanitize=address,undefined -fno-omit-frame-pointer
else
$(error BUILD must be portable, paper-native, or sanitize)
endif
CXXFLAGS := $(COMMON_FLAGS) $(PROFILE_FLAGS)
LDFLAGS := -shared -fopenmp
SANITIZER_HARNESS := build/sanitize/sanitizer_harness
SANITIZER_FLAGS := -std=c++20 -fopenmp -O1 -g -march=x86-64-v2 -fsanitize=address,undefined -fno-omit-frame-pointer
ifeq ($(BUILD),sanitize)
LDFLAGS += -fsanitize=address,undefined
endif

.PHONY: all clean FORCE manifest sanitize-check

all: $(TARGET)

$(TARGET): $(PROFILE_TARGET) FORCE
	cp $(PROFILE_TARGET) $(TARGET)

$(PROFILE_TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJECTS)
	python3 ../../scripts/native_build_manifest.py --binary $@ --compiler $(CXX) \
		--flags "$(CXXFLAGS) $(LDFLAGS)" --profile $(BUILD) --output $(MANIFEST)

$(PROFILE_DIR)/%.o: %.cpp $(HEADERS)
	mkdir -p $(PROFILE_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

manifest: $(MANIFEST)

$(MANIFEST): $(PROFILE_TARGET)

sanitize-check: $(SANITIZER_HARNESS)
	ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 $(SANITIZER_HARNESS)

$(SANITIZER_HARNESS): $(SOURCES) sanitizer_harness.cpp $(HEADERS)
	mkdir -p $(@D)
	$(CXX) $(SANITIZER_FLAGS) -o $@ $(SOURCES) sanitizer_harness.cpp

FORCE:

clean:
	rm -rf build $(TARGET) *.o

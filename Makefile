appname := nnd

sources := dtypes.cpp rp_trees.cpp nnd.cpp test.cpp utils.cpp
objects := $(patsubst %.cpp,%.o,$(sources))
depends := $(patsubst %.cpp,%.d,$(sources))

CXX := g++
CC_CLANG := clang++
CXXFLAGS := -Wall -g -O3 -pg -march=native
CXXFLAGS_O3 := -Wall -g -O3
EXTRA_FLAGS := -fopenmp
# CXXFLAGS := -Ofast -march=native

CFLAG_FAST := -Wall -g -Ofast -march=native -flto -fno-math-errno
CFLAG_DEBUG := -Wall -Wextra -g -Og
# CFLAG_DEBUG := -Wall -Wextra -g -O0 -pg -fno-stack-protector -fno-inline-functions

all: $(appname)

install:
	pip install -e .

$(appname): $(objects)
	$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) $^ -o $@

-include $(depends)

%.o: %.cpp Makefile
	$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) -MMD -MP -c $< -o $@
# $(CXX) $(CXXFLAGS) -MMD -c $<

clean:
	rm -fr *.o *.d *.h.gch $(appname) build *.so *.egg-info .eggs gmon.out gprof.png
	ctags -R .

run:
	./$(appname)

debug: CXXFLAGS := $(CFLAG_DEBUG)
debug: $(appname)

fast: CXXFLAGS := $(CFLAG_FAST)
fast: $(appname)

o3: CXXFLAGS := $(CXXFLAGS_O3)
o3: $(appname)

clang: CXX := $(CC_CLANG)
clang: CXXFLAGS := -Wall -g -Ofast -march=native -pg
clang: $(appname)
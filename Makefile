appname := nnd

sources := dtypes.cpp rp_trees.cpp nnd.cpp test.cpp utils.cpp
objects := $(patsubst %.cpp,%.o,$(sources))
depends := $(patsubst %.cpp,%.d,$(sources))

CXX := g++
CC_CLANG := clang++
CXXFLAGS := -Wall -g -pg -Ofast -march=native -flto -fno-math-errno
CXXFLAGS_O3 := -Wall -g -O3
EXTRA_FLAGS := -fopenmp -pg
# RIGHT_FLAGS := -ltbb
# CXXFLAGS := -Ofast -march=native

CFLAG_FAST := -Wall -g -Ofast -march=native -flto -fno-math-errno
CFLAG_DEBUG := -Wall -Wextra -g
CFLAG_DEBUG := -Wall -Wextra -g -O0 -pg -fno-stack-protector -fno-inline-functions

all: $(appname)

install:
	pip install -e .

$(appname): $(objects)
	$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) $^ -o $@ $(RIGHT_FLAGS)

-include $(depends)

%.o: %.cpp Makefile
	$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) -MMD -MP -c $< -o $@ $(RIGHT_FLAGS)
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

valgrind: CXXFLAGS := -Wall -g -O3 -march=native -pg
valgrind: $(appname)

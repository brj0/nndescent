appname=nnd

srcfiles=$(shell find . -name "*.cpp")
objects=$(patsubst %.cpp, %.o, $(srcfiles))
objects=dtypes.o nnd.o test.o utils.o
headers=dtypes.h nnd.h utils.h

CC=g++
CC_CLANG=clang++
CFLAG=-Wall -g -Ofast -pg -march=native
# CFLAG=-Ofast -march=native

CFLAG_FAST=-Ofast -march=native -flto -fno-math-errno
CFLAG_DEBUG=-Wall -Wextra -g -O0 -pg -fno-stack-protector

all: $(appname)

install:
	pip install -e .

$(appname): $(objects)
	$(CC) $(CFLAG) $^ -o $@

%.o: %.cpp %.h
	$(CC) $(CFLAG) -c $<

clean:
	rm -fr *.o $(appname) build *.so *.egg-info .eggs gmon.out gprof.png
	ctags -R .

run:
	./$(appname)

debug: CFLAG=$(CFLAG_DEBUG)
debug: $(appname)

fast: CFLAG=$(CFLAG_FAST)
fast: $(appname)

clang: CC=$(CC_CLANG)
clang: CFLAG=-Wall -g -Ofast -march=native
clang: $(appname)

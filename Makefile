appname=nnd

srcfiles=$(shell find . -name "*.cpp")
objects=$(patsubst %.cpp, %.o, $(srcfiles))
objects=dtypes.o nnd.o test.o utils.o
headers=dtypes.h nnd.h utils.h

CC=g++
CFLAG=-Wall -g -O3

all: $(appname)

install:
	pip install -e .

$(appname): $(objects)
	$(CC) $(CFLAG) $^ -o $@

%.o: %.cpp %.h
	$(CC) $(CFLAG) -c $<

clean:
	rm -fr *.o $(appname) build *.so *.egg-info .eggs
	ctags -R .

run:
	./$(appname)

debug: CFLAG=-Wall -Wextra -g -O0 -pg -fno-stack-protector

debug: $(appname)

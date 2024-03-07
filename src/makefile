CC = g++
CFLAGS = -std=c++14 -O3

SRCS = lsh.cpp point.cpp
OBJS = $(SRCS:.cpp=.o)

all: lsh

lsh: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

clean:
	rm -f $(OBJS) lsh

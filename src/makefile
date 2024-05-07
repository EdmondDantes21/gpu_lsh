CC = g++
CFLAGS = -fopenmp
TARGET = lsh
SRCS = lsh.cpp index.cpp point.cpp index_CPU.cpp

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)

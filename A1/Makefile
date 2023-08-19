CXX = g++
CXXFLAGS = -std=c++11 -Wall

TARGET = main
SRCS = main.cpp fptree.cpp
OBJS = $(SRCS:.cpp=.o)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean

run: main
	./main "/home/parth/A1_datasets/D_medium.dat" ./compressed.dat
# ./main ./test.dat ./compressed.dat

run_decompress: decompress
	./decompress ./compressed.dat ./decompressed.dat

decompress: decompress.o
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(OBJS) $(TARGET) decompress decompress.o

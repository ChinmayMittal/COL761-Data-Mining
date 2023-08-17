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
	./main "/Users/chinmaymittal/Downloads/A1_datasets/D_medium2.dat"
# ./main ./test.dat

clean:
	rm -f $(OBJS) $(TARGET)

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -Wall

# Source file
SOURCE = format_change.cpp

# Output executable
OUTPUT = format_change

all: $(OUTPUT)

$(OUTPUT): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(OUTPUT) $(SOURCE)

clean:
	rm -f $(OUTPUT)

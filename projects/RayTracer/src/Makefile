CXX = g++

CXXFLAGS = -std=c++17 -I./sdl/include -w

LDFLAGS = -L./sdl/lib -lSDL3

SRC = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)
TARGET = main.exe

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	del /Q $(OBJ) $(TARGET) 2>nul
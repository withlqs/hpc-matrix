CXX = icpc
CXXFLAGS += -O3 -Wall
SRC_DIR = src

generator.run: $(SRC_DIR)/generator/main.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

viewer.run: $(SRC_DIR)/viewer/main.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: clean matrix

matrix: generator.run
	./$< 500 a.mat 1994
	./$< 500 b.mat 1995

clean:
	-rm ./*.run
	-rm ./*.mat

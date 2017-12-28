CXX = icpc
CXXFLAGS += -O3 -Wall -Isrc -fopenmp -march=native
SRC_DIR = src
SIZE = 500

cpu.run: $(SRC_DIR)/cpu/main.cpp utils.o
	$(CXX) $(CXXFLAGS) $^ -o $@

generator.run: $(SRC_DIR)/generator/main.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

viewer.run: $(SRC_DIR)/viewer/main.cpp utils.o
	$(CXX) $(CXXFLAGS) $^ -o $@

utils.o: $(SRC_DIR)/utils/main.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

a.mat: generator.run
	./$< $(SIZE) $@ 1994

b.mat: generator.run
	./$< $(SIZE) $@ 1995

.PHONY: clean matrix test

test: cpu.run a.mat b.mat
	./$< a.mat b.mat c.mat


matrix: generator.run
	./$< $(SIZE) a.mat 1994
	./$< $(SIZE) b.mat 1995

clean:
	-rm ./*.run
	-rm ./*.mat
	-rm ./*.o

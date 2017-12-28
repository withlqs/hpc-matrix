CXX = icpc
CXXFLAGS += -O3 -Wall -Isrc -fopenmp -march=native
SRC_DIR = src

cpu.run: $(SRC_DIR)/cpu/main.cpp utils.o
	$(CXX) $(CXXFLAGS) $^ -o $@

generator.run: $(SRC_DIR)/generator/main.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

viewer.run: $(SRC_DIR)/viewer/main.cpp utils.o
	$(CXX) $(CXXFLAGS) $^ -o $@

utils.o: $(SRC_DIR)/utils/main.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

a.mat: generator.run
	./$< 1000 $@ 1994

b.mat: generator.run
	./$< 1000 $@ 1995

.PHONY: clean matrix test

test: cpu.run a.mat b.mat
	./$< a.mat b.mat c.mat


matrix: generator.run
	./$< 1000 a.mat 1994
	./$< 1000 b.mat 1995

clean:
	-rm ./*.run
	-rm ./*.mat
	-rm ./*.o

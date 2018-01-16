CXX = icpc
CXXFLAGS += -O3 -Wall -Isrc/headers -march=native
SRC_DIR = src
SIZE = 8000

cpu.run: $(SRC_DIR)/cpu/main.cpp common.o
	$(CXX) $(CXXFLAGS) -qopenmp $^ -o $@

generator.run: $(SRC_DIR)/generator/main.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

viewer.run: $(SRC_DIR)/viewer/main.cpp common.o
	$(CXX) $(CXXFLAGS) $^ -o $@

validator.run: $(SRC_DIR)/validator/main.cpp common.o
	$(CXX) $(CXXFLAGS) -mkl $^ -o $@

common.o: $(SRC_DIR)/common/main.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

a.mat: generator.run Makefile
	./$< $(SIZE) $@ 1994

b.mat: generator.run Makefile
	./$< $(SIZE) $@ 1995

.PHONY: view test_cpu clean

view: viewer.run a.mat
	./$< a.mat

test_cpu: cpu.run a.mat b.mat validator.run
	./$< a.mat b.mat c.mat
	./validator.run a.mat b.mat c.mat

clean:
	-rm ./*.run
	-rm ./*.mat
	-rm ./*.o

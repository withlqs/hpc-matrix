CXX = icpc
CXXFLAGS = -O3 -Wall -Isrc/headers -march=native
SRC_DIR = src
SIZE = 8000
SEED1 = 1994
SEED2 = 1995

NVCC = nvcc
NVCCFLAGS = -O3 -Isrc/headers -Xcompiler="-O3 -Wall -march=native" -Xlinker="-O3 -Wall"

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
	./$< $(SIZE) $@ $(SEED1)

b.mat: generator.run Makefile
	./$< $(SIZE) $@ $(SEED2)

gpu.run: $(SRC_DIR)/gpu/main.cu common.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

.PHONY: view test_cpu clean

view: viewer.run a.mat
	./$< a.mat

cpu: cpu.run a.mat b.mat validator.run
	srun --exclusive -p cpu ./$< a.mat b.mat c.mat
	srun --exclusive -p cpu ./validator.run a.mat b.mat c.mat

gpu: gpu.run a.mat b.mat validator.run
	srun --exclusive -p gpu ./$< a.mat b.mat c.mat
	srun --exclusive -p cpu ./validator.run a.mat b.mat c.mat

knl: cpu.run a.mat b.mat validator.run
	srun --exclusive -N 13 ./$< a.mat b.mat c.mat
	srun --exclusive -N 13 ./validator.run a.mat b.mat c.mat

clean:
	-rm ./*.run
	-rm ./*.mat
	-rm ./*.o

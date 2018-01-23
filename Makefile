SRC_DIR = src
SIZE = 8000
SEED1 = 1994
SEED2 = 1995

CXX = icpc
NVCC = nvcc

CXXFLAGS = -O3 -Wall -Isrc/headers
NVCCFLAGS = -O3 -Isrc/headers -Xcompiler="-O3 -Wall -march=native" -Xlinker="-O3 -Wall"
CPU_FLAGS = -xHost -march=native
KNL_FLAGS = -xmic-avx512

CPU_VALIDATOR = cpu_validator.run
GPU_VALIDATOR = gpu_validator.run
KNL_VALIDATOR = knl_validator.run
COMMON_LIB = common.o
COMMON_LIB_KNL = common_knl.o


generator.run: $(SRC_DIR)/generator/main.cpp
	$(CXX) $(CXXFLAGS) $(CPU_FLAGS) $^ -o $@

viewer.run: $(SRC_DIR)/viewer/main.cpp $(COMMON_LIB)
	$(CXX) $(CXXFLAGS) $(CPU_FLAGS) $^ -o $@

$(CPU_VALIDATOR): $(SRC_DIR)/validator/main.cpp $(COMMON_LIB)
	$(CXX) $(CXXFLAGS) $(CPU_FLAGS) -mkl $^ -o $@

$(GPU_VALIDATOR): $(SRC_DIR)/validator/main.cpp $(COMMON_LIB)
	#TODO
	$(CXX) $(CXXFLAGS) $(CPU_FLAGS) -mkl $^ -o $@

$(KNL_VALIDATOR): $(SRC_DIR)/validator/main.cpp $(COMMON_LIB_KNL)
	$(CXX) $(CXXFLAGS) $(KNL_FLAGS) -mkl $^ -o $@

$(COMMON_LIB): $(SRC_DIR)/common/main.cpp
	$(CXX) $(CXXFLAGS) $(CPU_FLAGS) -c $^ -o $@

$(COMMON_LIB_KNL): $(SRC_DIR)/common/main.cpp
	$(CXX) $(CXXFLAGS) $(KNL_FLAGS) -c $^ -o $@

a.mat: generator.run Makefile
	./$< $(SIZE) $@ $(SEED1)

b.mat: generator.run Makefile
	./$< $(SIZE) $@ $(SEED2)

cpu.run: $(SRC_DIR)/cpu/main.cpp $(COMMON_LIB)
	$(CXX) $(CXXFLAGS) $(CPU_FLAGS) -qopenmp $^ -o $@

gpu.run: $(SRC_DIR)/gpu/main.cu $(COMMON_LIB)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

knl.run: $(SRC_DIR)/knl/main.cpp $(COMMON_LIB_KNL)
	$(CXX) $(CXXFLAGS) $(KNL_FLAGS) -qopenmp $^ -o $@


.PHONY: view test_cpu clean

view: viewer.run a.mat
	./$< a.mat

cpu: cpu.run a.mat b.mat $(CPU_VALIDATOR)
	srun --exclusive -p cpu ./$< a.mat b.mat c.mat
	srun --exclusive -p cpu ./$(CPU_VALIDATOR) a.mat b.mat c.mat

gpu: gpu.run a.mat b.mat $(GPU_VALIDATOR)
	srun --exclusive -p gpu ./$< a.mat b.mat c.mat
	srun --exclusive -p cpu ./$(GPU_VALIDATOR) a.mat b.mat c.mat

knl: knl.run a.mat b.mat $(KNL_VALIDATOR)
	#TODO change ssh or srun and MCDRAM
	srun -exclusive -N 13 ./$< a.mat b.mat c.mat
	srun -exclusive -N 13 ./$(KNL_VALIDATOR) a.mat b.mat c.mat

clean:
	-rm ./*.run
	-rm ./*.mat
	-rm ./*.o

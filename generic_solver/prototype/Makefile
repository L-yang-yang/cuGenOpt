# GenSolver Prototype Makefile
# 用法:
#   make          → 编译 main
#   make run      → 编译并运行
#   make verify   → 编译并运行场景验证
#   make clean    → 清理

NVCC     = nvcc
ARCH     = -arch=sm_75
CFLAGS   = -O2 -std=c++17
INCLUDES = -I core -I problems

# 头文件依赖
CORE_HEADERS = core/cuda_utils.cuh core/types.cuh core/operators.cuh \
               core/population.cuh core/solver.cuh core/gpu_cache.cuh \
               core/init_selection.cuh
PROB_HEADERS = problems/tsp.cuh problems/tsp_large.cuh problems/tsp_xlarge.cuh problems/knapsack.cuh problems/assignment.cuh problems/schedule.cuh problems/vrp.cuh

# 主程序
main: main.cu $(CORE_HEADERS) $(PROB_HEADERS)
	$(NVCC) $(ARCH) $(CFLAGS) $(INCLUDES) -o main main.cu

# 性能基准测试
bench: bench_tsp51.cu $(CORE_HEADERS) $(PROB_HEADERS)
	$(NVCC) $(ARCH) $(CFLAGS) $(INCLUDES) -o bench bench_tsp51.cu

# 交叉效果对比
bench_cx: bench_crossover.cu $(CORE_HEADERS) $(PROB_HEADERS)
	$(NVCC) $(ARCH) $(CFLAGS) $(INCLUDES) -o bench_cx bench_crossover.cu

# 解质量验证（标准实例 vs 已知最优）
quality: bench_quality.cu $(CORE_HEADERS) $(PROB_HEADERS)
	$(NVCC) $(ARCH) $(CFLAGS) $(INCLUDES) -o quality bench_quality.cu

# 配置策略对比
cfgcmp: bench_config_compare.cu $(CORE_HEADERS) $(PROB_HEADERS)
	$(NVCC) $(ARCH) $(CFLAGS) $(INCLUDES) -o cfgcmp bench_config_compare.cu

# 快速回归验证（低资源）
quickv: bench_quick_verify.cu $(CORE_HEADERS) $(PROB_HEADERS)
	$(NVCC) $(ARCH) $(CFLAGS) $(INCLUDES) -o quickv bench_quick_verify.cu

# LNS 消融实验
ablation: bench_ablation_lns.cu $(CORE_HEADERS) $(PROB_HEADERS)
	$(NVCC) $(ARCH) $(CFLAGS) $(INCLUDES) -o ablation bench_ablation_lns.cu

# 大规模实例验证
largescale: bench_large_scale.cu $(CORE_HEADERS) $(PROB_HEADERS) problems/tsplib_data.h
	$(NVCC) $(ARCH) $(CFLAGS) $(INCLUDES) -o largescale bench_large_scale.cu

# 规模扩展测试（delta vs full evaluate）
scale: bench_tsp_scale.cu $(CORE_HEADERS)
	$(NVCC) $(ARCH) $(CFLAGS) $(INCLUDES) -o scale bench_tsp_scale.cu

# 场景验证（独立，不依赖框架）
verify: tests/verify_scenarios.cu tests/test_scenarios.h
	$(NVCC) $(ARCH) $(CFLAGS) -I tests -o verify tests/verify_scenarios.cu

# 运行
run: main
	./main

run-verify: verify
	./verify

run-bench: bench
	./bench

run-quality: quality
	./quality

# 清理
run-largescale: largescale
	./largescale

# 清理
clean:
	rm -f main verify bench bench_cx scale quality cfgcmp largescale quickv ablation *.o

.PHONY: run run-verify run-bench run-quality run-largescale clean

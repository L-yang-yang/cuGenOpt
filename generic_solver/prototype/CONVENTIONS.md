# GenSolver 原型工程约定

> 原则：够用就好。能在一个月后看懂自己的代码，但不为"可能的未来扩展"提前设计。

## 一、为什么 CUDA 项目的工程规范和 Java/Python 不同

```
传统 OOP 依赖：
  ① 虚函数 → 运行时多态 → GPU 上不能用（__device__ 函数无 vtable）
  ② 堆分配 → new/delete → GPU 上极其昂贵
  ③ 继承层次 → 增加间接寻址 → GPU 性能杀手

CUDA 行业标准（CUB / Thrust / cuOpt / RAPIDS）：
  ① 编译时多态（模板 + 概念约束）替代虚函数
  ② POD 结构体（Plain Old Data）替代类层次
  ③ 值语义（直接拷贝）替代引用语义
  ④ Host 端 RAII 管理 GPU 内存，Device 端纯算法无资源管理
```

## 二、我们的四条规则

### 规则1: 两层分离 — Host管资源，Device算数据

```
Host（CPU）职责:
  ✓ 分配/释放 GPU 内存 (cudaMalloc/cudaFree)
  ✓ 数据传输 (cudaMemcpy)
  ✓ 配置参数、启动 kernel、读取结果
  ✓ RAII 生命周期管理

Device（GPU）职责:
  ✓ 纯计算：评估、搜索、比较
  ✓ 接收指针和配置，不拥有任何资源
  ✓ 不做 malloc/free
```

**落地：Host 端用 class + 析构函数管 GPU 内存，Device 端全用 struct + free function。**

### 规则2: 编译时多态 — 模板参数传入"问题定义"

```cpp
// ❌ 错误：虚函数（GPU 不支持）
struct Problem {
    virtual __device__ void evaluate(/*...*/) = 0;
};

// ❌ 过度：CRTP + tag dispatch + enable_if
template<typename Derived, typename Tag, 
         std::enable_if_t<is_problem_v<Derived>>* = nullptr>
struct ProblemBase { ... };

// ✅ 正确：简单模板 + 约定接口（鸭子类型）
template<typename Problem>
__global__ void evaluate_kernel(Problem prob, Solution* pop, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) prob.evaluate(pop[tid]);
}
```

**Problem 是一个结构体，满足约定的接口即可。不需要继承任何基类。**

### 规则3: 配置用 struct，不用长参数列表

```cpp
// ❌ 参数太多
void solve(int pop_size, int max_gen, float mutation_rate, 
           int elite_count, unsigned seed, ...);

// ✅ 配置结构体
struct SolverConfig {
    int pop_size     = 128;
    int max_gen      = 1000;
    float mut_rate   = 0.1f;
    int elite_count  = 4;
    unsigned seed    = 42;
};
void solve(const SolverConfig& cfg, ...);
```

### 规则4: 错误不能静默 — CUDA 调用必须检查

```cpp
// 每次 CUDA API 调用都用这个宏包裹
#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)
```

## 三、命名约定

```
类型/结构体     PascalCase        Solution, SolverConfig, TSPProblem
函数（host）    snake_case        create_population, run_solver
函数（device）  snake_case        evaluate, apply_swap
Kernel         snake_case_kernel  evaluate_kernel, swap_search_kernel
常量/宏        UPPER_SNAKE        MAX_DIM1, CUDA_CHECK
枚举值         PascalCase        EncodingType::Permutation
文件名         snake_case.cuh    types.cuh, operators.cuh
GPU指针        d_ 前缀           d_population, d_dist_matrix
CPU指针        h_ 前缀(需要时)   h_best_solution
```

## 四、文件组织

```
prototype/
├── core/                      # 框架核心（问题无关）
│   ├── cuda_utils.cuh         #   CUDA_CHECK, 设备信息, curand wrapper
│   ├── types.cuh              #   Solution, SolverConfig, EncodingType
│   ├── operators.cuh          #   __device__ 算子: swap, reverse, insert, flip
│   ├── population.cuh         #   Population 类(Host RAII): 分配/初始化/精英
│   └── solver.cuh             #   solve<Problem>(): 主循环模板
├── problems/                  # 问题定义（"用户代码"示范）
│   ├── tsp.cuh                #   TSPProblem
│   ├── knapsack.cuh           #   KnapsackProblem（后续）
│   └── vrp.cuh                #   VRPProblem（后续）
├── tests/                     # 验证和测试
│   ├── test_scenarios.h       #   小实例验证集
│   └── verify_scenarios.cu    #   场景验证程序
├── main.cu                    # 入口
├── Makefile                   # 编译规则 (-I core -I problems)
└── CONVENTIONS.md             # 本文件
```

**三层分离：core/ 框架核心、problems/ 问题定义、tests/ 验证测试。**
**每个文件 50-200 行，总计 ~600-800 行。不会出现 1000 行的巨文件。**

## 五、要做 / 不做 对照表

```
✅ 要做                              ❌ 不做
─────────────────────────────────────────────────────────
POD struct + 自由函数(device)        class 继承层次
template<Problem> 编译时多态          virtual + vtable
SolverConfig 配置 struct              Builder 模式
CUDA_CHECK 包裹所有 API 调用         忽略返回值
Host 类管 GPU 内存 (析构 free)        手动配对 malloc/free
枚举区分编码类型                      字符串/magic number
const __restrict__ 标记只读指针       到处用 volatile
一个 .cuh = 一个职责                  一个文件放所有代码
注释解释 "为什么"                    注释复述代码
固定 MAX 上限 + 实际 size 字段        动态数组(GPU 上)
```

## 六、设计模式使用指南

```
适合 CUDA 的模式:
  ① 模板方法 (Template Method) — 但用真模板，不用虚函数
  ② 策略模式 (Strategy) — 用模板参数传入，编译时绑定
  ③ RAII — 仅 Host 端，管理 cudaMalloc/cudaFree
  ④ 配置对象 (Config Object) — 替代长参数列表

不适合的模式（别用）:
  ✗ 工厂模式 — GPU 不做运行时类型创建
  ✗ 观察者模式 — GPU 不做回调
  ✗ 单例模式 — 全局状态是 GPU 并行的噩梦
  ✗ 装饰器模式 — 需要间接寻址，性能杀手
```

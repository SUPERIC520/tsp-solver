# TSP 求解器重构与迁移需求文档 (proposal.md)

## 1. 背景与目标 (Background & Objective)
当前项目的大规模 TSP（旅行商问题）求解器在 Git 的 `exp` 分支中包含了许多实验性代码（例如不同的种子生成策略、不同的候选集构建方式、高阶 K-opt 以及混合的 C++ 尝试）。为了确保代码库的长期可维护性、高内聚性以及代码的干净整洁，我们需要将最核心、最有效的算法版本（Core Version）提取、提炼并迁移至 `main` 分支。

**核心交付目标**：
- 保证 100% 纯 Python + Numba 架构，去除所有不必要的实验性尝试。
- 在 `main` 分支中建立清晰、鲁棒且符合工程规范的代码结构，用于解决规模达 ~11.5 万个城市的 TSP 问题。
- 精简冗余的测试与基准代码，仅保留核心的功能单元测试。

---

## 2. 架构概述 (Architecture Overview)

### 2.1 技术栈约束
- **100% 纯 Python**：整个项目完全由 Python 编写，严禁引入或编译任何 C++ 代码。
- **Numba `@njit` 加速**：计算密集型逻辑（K-opt 邻域搜索、距离计算、数组位置更新、Held-Karp 迭代等）均使用 Numba 进行 `@njit(fastmath=True, cache=True)` 编译，以最大化 CPU 缓存局部性与执行效率。

### 2.2 核心算法与搜索策略
- **候选集构建（KD-Tree & Alpha 过滤）**：
  - **完全弃用 Delaunay 三角剖分**，仅使用 Scipy 的 `KDTree` 快速构建基础邻居集（如 Top 16）。
  - 结合 **Held-Karp (HK) 下界估计** 计算的 `Pi` 向量，评估每条边的 Alpha 值（LKH 估值法）。
  - 根据 Alpha 值从小到大重新排序候选集，优先搜索更有可能出现在最优解中的边。
- **级联 K-opt 搜索（Cascading K-opt）**：
  - 局部搜索的执行顺序严格为：**`2-opt` $\rightarrow$ `or-opt` $\rightarrow$ `3-opt`**（即在 2-opt 和 3-opt 之间插入 `or-opt` 算子）。
  - **搜索剪枝**：结合 Don't Look Bits (DLB) 标志数组，跳过未修改节点的冗余扫描；并在搜索中注入软骨架（Soft Backbone）提取出的共识锁定边（Locked Edges）以大幅收窄搜索分支。
- **播种与重播种策略（Seeding & Reseed）**：
  - **初始播种 (Initial Seeding)**：仅使用高效的**贪心算法 (Greedy NN)** 生成初始路径，弃用 Hilbert 曲线或随机扰动等其它播种机制（Hilbert 曲线仅用于城市坐标重排以优化 cache locality）。
  - **重播种续航 (Reseed continuation)**：采用纯**开发/利用 (Exploit)** 策略，即在迭代中选取当前全局最优路径（Best Tour），对其进行**旋转 (Rotation)** 变换后作为下一轮优化的初始态。
- **并行处理**：
  - 使用 Python 的 `multiprocessing.Pool` 在多核 CPU 上并行处理多个种子实例。
  - 保持现有 `exp` 分支的多进程并发逻辑不变，但需增强其进程管理和异常捕获的鲁棒性。

---

## 3. 核心迁移与剥离清单 (Migration Checklist)

| 模块/文件 | 迁移与提炼内容 (Keep & Refine) | 舍弃与剥离内容 (Discard & Remove) |
| :--- | :--- | :--- |
| **工程依赖 & 技术栈** | - 纯 Python 依赖（`numpy`, `scipy`, `numba`）<br>- Numba `@njit` 编译 | - 任何 C++ 代码与绑定依赖（pybind11、ctypes、C++ 编译脚本等） |
| **全局配置管理** | - 新增 `src/config.py`，将“动态漏斗 (Dynamic Funneling)”等所有硬编码参数抽取为全局配置 | - 剥离各核心算法文件中硬编码的搜索深度、限制条件和漏斗参数 |
| **候选集生成** | - 使用 `scipy.spatial.KDTree` 生成候选集<br>- 使用 Held-Karp `Pi` 向量计算 Alpha 值并重排候选集 | - 彻底移除基于 Delaunay 三角剖分的候选集生成逻辑<br>- 移除 Delaunay 与 KDTree 的混合过滤算法 |
| **种子生成策略** | - 基于候选集的快速贪心 Nearest-Neighbor (Greedy NN) 初始路径生成<br>- 针对当前 Best Tour 的旋转 (Rotation) 重播种机制 | - 移除 Hilbert 曲线生成多样性初始种子的逻辑<br>- 移除随机 permutation 播种逻辑 |
| **K-opt 优化引擎** | - `2-opt` -> `or-opt` -> `3-opt` 级联搜索<br>- DLB (Don't Look Bits) 与 Soft Backbone (Locked Edges) 剪枝<br>- 基础 Double-Bridge Kick 扰动 | - 移除 4-opt 和 5-opt 的相关实验性函数代码<br>- 移除所有不符合 2-opt -> or-opt -> 3-opt 顺序的级联逻辑<br>- 移除重复/冗余的优化引擎版本（如 `lkh_core.py`） |
| **并发调度模块** | - 并行多起点执行逻辑 `parallel_solve` | - 增加鲁棒的 `try-except` 异常捕获以避免子进程挂死导致的整个求解器挂起 |
| **测试与基准** | - 核心功能的单元测试（2-opt, 3-opt, or-opt 正确性，Held-Karp 计算等） | - 彻底移除所有针对大规模数据的性能基准测试脚本（如 `run_benchmarks.py`） |

---

## 4. 任务拆解与后续步骤 (Tasks & Next Steps)

### 第一阶段：配置抽离与环境清理
1. **[NEW] 创建 [config.py](file:///C:/Users/eric2/Desktop/Classes/Math%20147/TSP_EXP_2/src/config.py)**：
   - 提取参数：候选集大小 `K_NEIGHBORS` (默认 16)、级联最大阶段、Or-opt 的最大插入长度 `OR_OPT_MAX_LEN` (默认 5)、软骨架判定阈值 `BACKBONE_THRESHOLD` (默认 0.95)、多进程核心数 `NUM_PROCESSES` 等。
2. **清理冗余文件**：
   - 删除 `src/core/lkh_core.py` 等实验性替代引擎。
   - 删除大文件基准测试代码（例如 `run_benchmarks.py`）。

### 第二阶段：预处理与种子生成模块重构
3. **重构 [preprocessing.py](file:///C:/Users/eric2/Desktop/Classes/Math%20147/TSP_EXP_2/src/core/preprocessing.py)**：
   - 删除 `_filter_nearest_neighbors` 中所有涉及 Delaunay 邻居的代码，重写为纯 KD-Tree 查询。
   - 确保 `refine_candidate_set_with_alpha` 的接口清晰，仅依据 Held-Karp 的 Alpha 值重新调整 KD-Tree 候选集内元素的顺序。
4. **重构 [seed_generation.py](file:///C:/Users/eric2/Desktop/Classes/Math%20147/TSP_EXP_2/src/core/seed_generation.py)**：
   - 保留 `generate_greedy_nn_seeds` 作为唯一的初始种子生成器。
   - 新增路径旋转 `rotate_tour(tour, start_node)` 辅助函数，确保可以将最优路径的任一起点旋转到数组首位，完成 Exploit 策略下的重播种。

### 第三阶段：核心引擎与调度重构
5. **重构 [kopt_engine.py](file:///C:/Users/eric2/Desktop/Classes/Math%20147/TSP_EXP_2/src/core/kopt_engine.py)**：
   - 清理不符合级联顺序的代码。在 `_full_cascade` 中，只按照 `2-opt` -> `or-opt` -> `3-opt` 的级联方式顺序执行。
   - 将 Hardcoded 参数（如 Or-opt 搜索长度、Stagnation Limit 等）替换为从 `src.config` 中导入。
6. **重构 [orchestration.py](file:///C:/Users/eric2/Desktop/Classes/Math%20147/TSP_EXP_2/src/core/orchestration.py)**：
   - 保持多进程调度，并在 worker 级加入完善的异常捕获与日志记录，确保任何 Numba 异常或意外错误不会使主进程死锁。

### 第四阶段：测试整合与验证
7. **重构测试目录 `tests/`**：
   - 确保只保留验证 2-opt、3-opt、or-opt 算子正确性，KDTree 生成，Held-Karp 精度计算等功能性单元测试。
8. **运行验证程序**：
   - 执行单元测试确保重构无 regressions。
   - 运行 N=100 及 N=1000 样例检查，确认重构后的核心级联求解器可正常收敛并输出正确的旅行商路径。

# TSP Solver 详细设计文档

## 1. 引言
本文档基于需求文档 (proposal.md) 和概要设计文档 (high-level-design.md)，为解决约 11.5 万个城市的旅行商问题（TSP）提供详细设计方案。本系统采用全级联 K-Opt (Full Cascading K-Opt) 启发式算法，包含 2/3/4/5-opt 变换，结合多进程并发与 Numba-JIT 的向量化/并行计算能力，力求在合理时间内获得与 Held-Karp 下界差距在 < 5% 以内的近似最优解。

## 2. 设计原则
1. **模块独立性**：各模块职责单一，接口定义清晰，均可独立进行单元测试。
2. **数据结构高效性**：各模块间通信主要依托一维或二维 NumPy 数组 (`float64`, `int32`)，以便于 Numba 编译与零拷贝传递。
3. **混合并行策略**：
   - **多起点并行 (Multi-start)**：使用 Python `multiprocessing.Pool` 实现进程级并行，在不同 CPU 核心上独立运行基于不同种子（Seed，最多 8 个）的全级联 K-Opt 优化。
   - **底层数据并行**：使用 Numba `@njit(parallel=True)` 配合 `prange` 加速计算密集型任务（如生成候选集、计算局部距离）。
   - **核心串行与剪枝**：K-Opt 的边缘交换过程严格保持串行，利用欧氏度量的三角不等式进行早期剪枝，以确保算法效率。
4. **强制 Sub-Agent 委托**：所有功能开发、测试与基准测试必须通过专门的 Sub-Agent 执行。
5. **严谨测试流程**：每一项新测试必须从 N=100 开始，成功后方可增加样本量。

---

## 3. 核心模块详细设计

### 3.1 数据 I/O 模块 (Data I/O Module)
**职责**：处理坐标数据的加载和最终路径结果的保存。同时负责 Held-Karp 下界与 Pi 向量的持久化缓存。

- **数据结构**：
  - 坐标矩阵 `coords`：`np.ndarray`，形状 `(N, 2)`，类型 `np.float64`。

### 3.2 预处理模块 (Preprocessing Module)
**职责**：基于 Delaunay 三角剖分计算每个城市的候选邻居集（Top 16）。

- **数据结构**：
  - 候选集矩阵 `candidate_set`：`np.ndarray`，形状 `(N, 16)`，类型 `np.int32`。存储每个城市由 Delaunay 提取出的最近 16 个邻居的索引。

### 3.3 种子生成模块 (Seed Generation Module)
**职责**：为最多 8 个优化实例提供多样性的初始路径（Seeds）。

- **数据结构**：
  - 种子集合 `seeds`：`np.ndarray`，形状 `(8, N)`，类型 `np.int32`。

### 3.4 全级联 K-Opt 优化引擎模块 (Full Cascading K-Opt Engine)
**职责**：实现通用的全级联 K-Opt 算法（包含 2, 3, 4, 5-opt），对单一初始解进行局部搜索优化。采用级联逻辑，优先尝试低阶交换，失败后再尝试高阶。**注意：初期测试阶段仅启用至 3-opt 以平衡速度。**

- **核心函数**：
  ```python
  @njit(fastmath=True)
  def cascading_kopt_optimize(initial_tour: np.ndarray, coords: np.ndarray, candidate_set: np.ndarray, locked_edges: np.ndarray) -> tuple[np.ndarray, float]:
      """
      全级联 K-Opt 核心逻辑。顺序尝试 2, 3, 4, 5 边交换。
      利用三角不等式进行剪枝。
      """
      pass
  ```

### 3.5 并行调度模块 (Orchestration Module)
**职责**：多核 CPU 并行调度，分配最多 8 个种子。

### 3.6 骨干共识模块 (Backbone Consensus Module)
**职责**：分析并行执行得到的多个优秀解，锁定高频出现的边（>95%）。

### 3.7 验证模块 (Validation Module)
**职责**：计算 Held-Karp 下界。支持缓存机制，避免重复计算。

---

## 4. 数据流向与执行时序 (Data Flow & Sequence)
1. **启动阶段**：读取 `cities.csv`。
2. **预处理阶段**：基于 Delaunay 三角剖分生成 `candidate_set` (Top 16)。
3. **种子生成**：获取最多 8 条初始 `seeds`。
4. **迭代优化循环**：
   - 调用 **并行调度模块**：派生多个 Worker，通过多进程将种子分发给 **全级联 K-Opt 优化引擎**。
   - 所有子进程执行完毕，回收优化后的路径。
   - 调用 **骨干共识模块**：更新 `locked_edges`。
5. **验证与输出阶段**：比对最终结果与 HK 下界，输出报告。

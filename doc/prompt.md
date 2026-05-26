# Autonomous TSP Solver Refactoring Workflow: Master Agent Instructions

本文件作为主 Agent (Master Agent) 的“行为宪法”与核心系统指令，用以驱动后续无人值守、全自动闭环的 TSP 求解器重构与迁移工作。主 Agent 必须严格遵守以下所有运作规程、质量门控与分支管理策略。

---

## 1. 角色定义与协作协议 (Role Definitions & Collaboration Protocol)

在整个重构生命周期中，采用“主-子代理 (Master-Worker)”双层架构进行协作。

### 1.1 主 Agent (Master Agent) - 全局监工与状态机
*   **职责**：
    1. **进度管理**：解析 `doc/tasks/progress.md`，决定当前执行的原子任务。
    2. **生命周期控制**：管理 Git 分支、暂存、提交、回滚，以及外部命令（如 Linter、Type checker、Test suite）的调度。
    3. **Worker 编排**：动态生成并指派 Worker Agent，向其注入任务背景与上下文。
    4. **质量门控判定**：执行 `pytest`、`mypy`、`ruff`，并决定是推进进度还是触发回滚熔断。
*   **限制**：不直接编写业务代码，仅通过指令协调 Worker 进行编码。

### 1.2 Worker Agent (子 Agent) - 最小功能开发者
*   **职责**：
    1. **任务聚焦**：专注于主 Agent 派发的单一原子任务（例如：实现某一个函数或修复特定测试）。
    2. **局部编码与自测**：在指定的源文件和测试文件中编写代码，并在局部运行单元测试确保语法与逻辑正确。
    3. **根因分析 (RCA)**：在触发第 3 次重试时，必须输出 Root Cause Analysis，剖析前两次失败的深层技术原因。
*   **限制**：禁止执行任何 Git Commit 或分支切换操作；只能修改与当前任务直接相关的文件。

---

## 2. 分支锁定与原子提交机制 (Branch Guardrails & Commit Alignment)

### 2.1 Git 分支隔离
*   **分支锁定**：所有开发、重构与修复工作必须**且只能**在 `main` 分支上进行。
*   **启动自检**：主 Agent 启动后的第一步，必须执行以下指令并检查输出：
    ```powershell
    git checkout main
    git status
    ```
    主 Agent 必须确认当前处于 `main` 分支，且工作区是干净的（没有未提交的修改）。如果不满足，必须立即中止并报错。
*   **只读 `exp` 分支**：`exp` 分支仅作为历史代码的参考库。主 Agent 或 Worker 可以通过类似 `git show exp:<filepath>` 或读取 `exp` 分支文件来借鉴已有的实现，但**严禁向 `exp` 分支提交任何修改，严禁执行 `git merge exp` 或 `git rebase exp`**。

### 2.2 任务状态与 Git Commit 原子对齐
为了确保 `main` 分支的版本历史清晰且具备高可追溯性，每一个原子任务（如 `[T1.1]`）的完成必须与 Git Commit 形成原子绑定：
1. **任务状态标记**：当且仅当一个任务通过了所有质量门控（第 4 节）后，主 Agent 修改对应的进度文件：
    *   在 `doc/tasks/progress.md` 中将该任务标记为 `[x]`。
    *   在对应的模块任务文件（如 `doc/tasks/config.md`）中标记为 `[x]`。
2. **原子性 Commit**：主 Agent 将**代码文件修改**与**上述 Markdown 进度文件修改**一同添加至暂存区，执行单次提交：
    ```powershell
    git add src/ doc/tasks/
    git commit -m "feat(tsp): [TaskID] <Module Name>: <Brief Description>"
    ```
    *示例*：
    ```powershell
    git commit -m "feat(tsp): [T1.1] Config: Initialize global configuration parameters"
    ```

---

## 3. 全自动闭环工作流 (Loop Execution Workflow)

主 Agent 必须在无人值守模式下，按照以下循环逻辑自动迭代执行：

```
[读取任务] ──> [指派 Worker] ──> [代码实现] ──> [质量门控校验] 
     ▲                                                │
     │                                     ┌──────────┴──────────┐
     │                                  [通过]                [失败]
     │                                     │                     │
[原子 Commit] ◄────────────────────────────┘              [进入自适应迭代修复]
                                                                 │
                                                          (重试 <= 3次)
                                                                 ├─► 成功 ──► [原子 Commit]
                                                                 └─► 失败 ──► [Git 回滚 & 挂起]
```

### 详细步骤：
1.  **扫描任务**：读取 `doc/tasks/progress.md`，找到排序最靠前的未完成任务 `[T_X.Y]`。
2.  **提取规格**：读取 `doc/tasks/` 下对应的模块任务 Markdown 文件，获取该任务的输入、输出、影响文件及实现细节。
3.  **生成 Worker 任务**：调用 Worker Agent，使用“第 6 节：上下文注入模板”拼装当前任务的 Prompt。
4.  **Worker 执行**：Worker Agent 修改目标代码，并在其局部运行快速校验。
5.  **收集并运行门控**：Worker 返回修改完毕的信号后，主 Agent 接管，在 `main` 分支全局运行**三道质量门控**。
6.  **门控结果分流**：
    *   **通过**：执行原子 Commit（第 2.2 节），然后回到步骤 1 处理下一个任务。
    *   **失败**：进入“第 4.2 节：自适应迭代修复机制”。

---

## 4. 质量门控与自适应修复机制 (Quality Gates & Recovery)

### 4.1 三道质量门控 (Three Quality Gates)
任何新写的模块或修改的代码，必须**无条件**通过以下三项本地校验：
1.  **pytest 单元测试**：
    ```powershell
    uv run pytest tests/
    ```
    所有单元测试（包括断言、边界条件、异常捕获测试）必须 100% 通过。
2.  **mypy 静态类型检查**：
    ```powershell
    uv run mypy --strict src/
    ```
    所有类型标注必须完全符合 Python 强类型规范，不得包含任何隐式或未处理的 `Any` 类型错误。
3.  **ruff 代码风格与质量检测**：
    ```powershell
    uv run ruff check src/
    ```
    不得出现任何未修复的代码风格、规范或潜在 Bug 告警。

### 4.2 自适应迭代修复与根因分析 (RCA) 机制
若在执行上述质量门控时发生任何失败（如测试报错、mypy/ruff 检查未通过）或 Worker 自身抛出执行异常，主 Agent 必须按照以下步骤进行自适应修复，最大重试次数为 **3 次**：

1.  **记录失败现场**：主 Agent 截获并记录详细的 Traceback 信息、编译报错以及 Linting/Typing 错误日志。
2.  **重试计数 (Counter < 3)**：
    *   主 Agent 将失败日志作为“负反馈上下文”重新组装 Task Prompt。
    *   重新指派 Worker Agent 进行代码修正。
3.  **第 3 次重试（强制上下文增强与 RCA 注入）**：
    *   若前 2 次修复尝试均宣告失败，主 Agent 在发起第 3 次重试时，必须向 Worker 注入**强制约束指令**：
        > "WARNING: This is your 3rd and final attempt to fix this issue. You are REQUIRED to perform a Root Cause Analysis (RCA) before writing any code. You must output a section analyzing:
        > 1. Why the previous implementation failed.
        > 2. The mismatch between requirements and current types/Numba JIT compilation.
        > 3. Your explicit fix strategy.
        > After outputting the RCA, provide the corrected code."
    *   Worker Agent 必须遵循此逻辑，先进行 RCA 分析再输出代码。
4.  **最终失败熔断回滚**：
    *   如果第 3 次重试后，质量门控依然无法全部通过，主 Agent 必须立即触发**质量门控熔断**。
    *   主 Agent 自动执行 Git 硬回滚，将 `main` 分支恢复至最近一次稳定 Commit 状态：
        ```powershell
        git reset --hard HEAD
        git clean -fd
        ```
    *   在进度文件中将当前任务标记为 `[FAILED]`，保存失败日志，**立刻停止运行并向用户报错，等待人工干预，严禁继续强行推进后续任务**。

---

## 5. 自动化测试与超时控制规则 (Timeouts & Enforcements)

### 5.1 命令级超时保护
为了防止 Numba 在 C-level 编译或执行过程中由于指针越界、多进程死锁、JIT 无限循环导致的终端挂起，主 Agent **在执行任何终端命令时必须显式配置 Timeout 参数**。一旦超时，必须立刻发送 `SIGKILL` 终止进程并抛出异常。

### 5.2 动态超时上限表
超时阈值已结合 Numba 首次运行的冷编译开销（内含 25% 缓冲时间）和 TSP 算法复杂度进行了精细校准，具体执行指令与超时限制如下：

| 校验类型 / 数据规模 | N = 100 | N = 500 | N = 1,000 | N = 5,000 & 10,000 | N = 115,475 (Full Scale) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`ruff` 静态分析** | 10s | 10s | 10s | 12s | 15s |
| **`mypy --strict` 类型检查** | 10s | 10s | 10s | 12s | 15s |
| **`pytest` 单元测试** | 30s (首次冷编译) | 10s | 15s | 45s | 90s |
| **优化求解器运行** | 15s | 10s | 25s | 150s | 600s (10分钟) |

*注：在执行 `pytest` 时，若捕获到超时，主 Agent 需按“第 4.2 节”处理。*

### 5.3 Held-Karp Bound 缓存机制
在运行涉及 Held-Karp 下界估值的验证任务时，为了避免在多次重试和迭代中重复进行高昂的 subgradient 计算，主 Agent 和 Worker 必须遵守以下缓存规则：
*   **缓存位置**：项目根目录下的 `.cache/hk_bounds.json`。
*   **逻辑流**：在运行评估脚本或测试前，必须读取该缓存。若存在对应城市规模 $N$ 的已计算 bound 及其 $Pi$ 向量，则直接 load 载入，禁止重复计算。仅在首次运行无缓存数据时才执行完整的 Held-Karp 计算并保存。

---

## 6. 上下文注入模板 (Context Injection Template)

主 Agent 在唤起 Worker Agent 执行具体原子任务时，必须填充并发送以下 Prompt 模板：

```markdown
# Role & Goal
You are a Worker Agent. Your goal is to implement/refactor task [TaskID] in main branch.

# System & Architectural Constraints (Non-negotiable)
1. Language: 100% Pure Python + Numba (@njit) only.
2. Dependencies: numpy, scipy, numba. Strictly NO compiled C++ extensions (pybind11, ctypes, etc.).
3. Memory: Under 16GB. NEVER instantiate any N*N distance matrix for large N. Keep candidate sets to K=16/40.
4. Precision: Coordinates must use np.float64, indices must use np.int32.
5. Compilation: Numba functions must be C-contiguous. Ensure `np.ascontiguousarray()` is used where appropriate.
6. Reference Material: You can reference exp branch code via `git show exp:<path_to_file>` for implementation ideas, but do not write directly to or merge from exp.

# Current Task Target
- Task ID: [Insert TaskID, e.g. T2.3]
- Module: [Insert Module Name, e.g. Preprocessing]
- Description: [Insert specific requirement description from doc/tasks/<module>.md]
- Files to Edit: 
  * [File Path 1] (Ensure to keep existing irrelevant code intact)
  * [File Path 2]

# Test & Debugging Context
- Active Target Tests: `pytest [path_to_test_file]::[test_name]`
- Previous Failures / Error Traceback (if any):
```[Insert detailed traceback or error log here]```
[Include RCA Instruction if this is Attempt #3: "WARNING: This is your 3rd and final attempt. You MUST output a Root Cause Analysis (RCA) explaining the cause of previous errors before providing code."]

Please implement the requested changes, ensure all type definitions are strict and clean, and report back with your modified file snippets.
```

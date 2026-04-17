# SSA vs Others WSI Agent 汇报稿

## 1. 项目目标

当前我们在做的是一个面向结直肠病理 `SSA vs others` 任务的 WSI Agent 原型，目标是模仿论文 **CPathAgent** 的三阶段逻辑，把病理医生在整张切片上的诊断过程拆成：

1. `Trace Agent`
2. `Navigate Agent`
3. `Observe & Reasoning Agent`

首版目标不是完整复现论文训练体系，而是先把一个 **工程上可跑通、可解释、可回放** 的大图分析流程搭起来，便于后续逐步替换成更强的多模态模型或 API backend。

当前任务固定为：

- 输入：`Adenoma_yx` 的 WSI
- 标签：`adenoma_yx_labels.csv`
- 代理任务：`SSA vs others`
- 输出：区域筛查结果、导航轨迹、多倍率观察日志、英文病理报告、最终二分类判断

## 2. 当前系统整体流程

### 2.1 总流程

```text
WSI
  -> overview / thumbnail
  -> Trace Agent
  -> TraceCluster {l_k, s_k, d_k}
  -> Navigate Agent
  -> trajectory {(x, y, m, o)}
  -> Observe & Reasoning Agent
  -> step logs + checklist report
  -> final SSA vs others prediction
  -> audit / replay / evaluation
```

### 2.2 各阶段含义

#### A. Trace Agent

目的：

- 在低倍 overview 上先筛出值得看的黏膜区域
- 将单纯 ROI 框升级成更接近论文形式的结构化区域簇

当前实现：

- 先做低层视觉预筛：背景过滤、黏膜候选、区域聚合
- 再给每个区域簇打上：
  - `l_k`: 区域语义标签
  - `s_k`: SSA 诊断优先级
  - `d_k`: 是否需要高倍复查

当前首版语义标签：

- `ssa_suspicious_mucosa`
- `non_ssa_mucosa`
- `background`
- `artifact`

当前 SSA 判据重点：

- 结构相关：
  - basal dilatation
  - crypt branching
  - horizontal growth
  - boot/L/T-shaped crypt
- 表面相关：
  - serration to base
  - mucus cap
  - abnormal maturation

#### B. Navigate Agent

目的：

- 针对每个高优先级区域簇，生成类似病理医生“先看低倍、再决定是否放大”的浏览路径

当前输出格式严格固定为：

```text
(x, y, m, o)
```

其中：

- `x, y`：level-0 坐标中心
- `m`：倍率，当前使用 `1.0x -> 2.5x -> 5.0x`
- `o`：本步的 `need_to_see`

当前策略：

- 按 `s_k` 排序
- 若 `d_k=true`，自动进入多倍率复查
- `o` 会明确写本步想确认的诊断目标，例如：
  - 表面锯齿延伸
  - mucus cap
  - basal contour
  - crypt base architecture

#### C. Observe & Reasoning Agent

目的：

- 按轨迹裁剪多倍率视野
- 逐步记录 “看到什么 -> 为什么重要 -> 下一步看什么”
- 最后给出英文病理报告和 `SSA vs others` 判断

当前每一步输出：

- `Observation`
- `Reasoning`
- `Next step`
- `criteria_hits`

最终输出：

- 英文 `Pathological Report`
- checklist 风格的证据汇总
- 最终 `SSA vs others` judgement

## 3. 当前已经实现到什么程度

### 3.1 已完成内容

- 已建立独立工程目录：`adenoma_agent/`
- 已将标签源切换为与 manifest 完全对齐的 `adenoma_yx_labels.csv`
- 已把任务固定为 `SSA vs others`
- 已将旧版 bbox-only Trace 升级为 `TraceCluster`
- 已将旧版导航升级为 `(x, y, m, o)` 轨迹
- 已将旧版 patch 描述升级为 step-level reasoning + 英文报告
- 已补充 replay、batch eval、pilot subset 构建能力
- 已保留 provider-agnostic backend 接口，后续可以接：
  - 外部 API
  - 本地 Patho-R1
  - heuristic fallback

### 3.2 当前真实 smoke 已跑通

已完成真实单例 smoke：

- case: `138189_751666001`
- label: `Hyperplastic polyps`
- binary target: `others`

当前真实输出包括：

- `TraceCluster`
- 多倍率导航路径
- step-level observation/reasoning
- checklist 报告
- 最终预测
- replay 日志
- 自动评测摘要

## 4. 一个真实样例的结果说明

### 4.1 Trace 阶段

当前 smoke 样例在 overview 上筛出了 `1` 个高风险区域簇：

- `l_k = ssa_suspicious_mucosa`
- `s_k = 3`
- `d_k = true`

说明：

- 该区域被系统判断为“需要进入更高倍复查的可疑黏膜区”
- 当前 reasoning 依据主要来自：
  - 黏膜区域保留
  - 与 route C screening hint 有较高重叠
  - 区域大小适合聚焦隐窝复查

### 4.2 Navigate 阶段

当前真实导航轨迹已经形成三步有效观察：

1. `1.0x`：总览黏膜区域，找 serrated surface / mucus cap / hotspot
2. `2.5x`：看表面锯齿延伸、crypt distribution、basal contour
3. `5.0x`：看 crypt base，确认 basal dilatation / branching / boot-L-T shape

这说明当前系统已经具备“从低倍到高倍”的基本临床浏览逻辑，而不是随机裁 patch。

### 4.3 Observe & Reasoning 阶段

当前系统能输出 checklist 风格的英文病理报告。

在这个样例中，系统记录到：

- supporting:
  - `serration_to_base`
  - `mucus_cap`
  - `abnormal_maturation`
- uncertain:
  - `basal_dilatation`
  - `crypt_branching`
  - `horizontal_growth`
  - `boot_l_t_shaped_crypt`

最终判断：

- `others`
- score = `0.48`

这个结果和当前样例真实标签 `Hyperplastic polyps` 一致。

## 5. 当前结果怎么理解

### 5.1 当前已经证明的事情

- 这套三阶段 agent 流程在本地是可以真实跑通的
- 系统已经不是简单的 patch classifier，而是有：
  - 区域筛查
  - 多倍率导航
  - 逐步推理
  - 最终英文报告
- 当前产物已经适合做汇报、做回放、做人工审核

### 5.2 当前还不能过度解读的地方

- 当前 smoke 主要还是 heuristic backend 在兜底，不是论文级多模态模型效果
- 当前 `1` 个 case 跑通只能说明流程正确，不能说明模型性能已经可靠
- 当前 checklist 中很多关键 SSA 隐窝结构还是 `uncertain`
- 还没有完成 `10-30` 例人工复核闭环
- 还没有做稳定的批量评测和失败案例分析

换句话说：

- **目前证明的是“系统框架已经成型”**
- **还没有证明“诊断性能已经足够好”**

## 6. 当前可直接展示的文件

### 6.1 汇报时建议展示

- 项目说明：
  - `adenoma_agent/README.md`
  - `adenoma_agent/SCOPE.md`
- 真实样例结果：
  - `adenoma_agent/artifacts/runs/smoke/138189_751666001/trace/trace_clusters.json`
  - `adenoma_agent/artifacts/runs/smoke/138189_751666001/navigation/navigation_steps.json`
  - `adenoma_agent/artifacts/runs/smoke/138189_751666001/observe/pathological_report.json`
  - `adenoma_agent/artifacts/runs/smoke/138189_751666001/case_result.json`
  - `adenoma_agent/artifacts/runs/smoke/138189_751666001/replay.md`
- 自动评测：
  - `adenoma_agent/artifacts/runs/smoke/evaluation_summary.json`
  - `adenoma_agent/artifacts/runs/smoke/case_predictions.csv`

### 6.2 当前 smoke 的自动评测摘要

- `case_count = 1`
- `ssa_binary_accuracy = 1.0`
- `ssa_specificity = 1.0`
- `avg_trajectory_length = 3.0`
- `avg_report_checklist_completeness = 1.0`

注意：

- 这里只有单例 smoke，不能当正式性能结论

## 7. 下一步需要做的事情

### 第一优先级：把“能跑”变成“更可信”

1. 打开更强 backend
- 当前 `external_command` 和 `local_patho_r1` 默认关闭
- 下一步要优先接通至少一个更强的多模态 backend
- 目标是让 Trace / Observe 不再主要依赖 heuristic

2. 做一个小规模 pilot 子集
- 先基于 `SSA vs others` 构建一个稳定 pilot，例如 `12` 个正类 + `12` 个负类
- 对这批样本固定清单、固定输出目录、固定评测方式

3. 跑一轮 batch 推理
- 在 pilot 子集上批量运行
- 输出：
  - case predictions
  - confusion matrix
  - replayable logs
  - failure cases

### 第二优先级：把“像论文”变成“更像临床”

4. 强化 SSA 专用 prompt
- 当前 `need_to_see` 和 report 已经围绕 SSA checklist
- 下一步应进一步强化临床术语和判断层级：
  - supporting
  - opposing
  - uncertain
  - insufficient evidence

5. 优化 Trace 的临床指向性
- 当前 Trace 已能输出 `ssa_suspicious_mucosa`
- 下一步可以继续增强：
  - 黏膜区域细分
  - 更好的 mucus-like 区域检测
  - 更稳定的 crypt-risk prioritization

6. 让导航更像医生而不是固定模板
- 当前导航已经是低倍 -> 中倍 -> 高倍
- 下一步可加入：
  - 根据 Observe 结果动态回访
  - 某一步发现 evidence 不足时切换到邻近区域
  - 更真实的自回归决策

### 第三优先级：形成正式汇报和论文式材料

7. 做 `10-30` 例人工复核
- 让人工判断：
  - 系统看的位置对不对
  - 轨迹顺序合不合理
  - 报告是不是围绕 SSA 判据
  - 最终结论是否可接受

8. 整理失败案例库
- 至少分为：
  - 背景/黏液干扰
  - 表面特征支持但隐窝结构不足
  - 高倍下证据不够
  - Trace 找错区域

9. 准备下一版汇报材料
- 一页系统流程图
- 两页真实 case replay
- 一页 pilot 评测摘要
- 一页 failure analysis

## 8. 汇报时可以怎么概括

可以用下面这段话直接口头汇报：

> 我们现在已经把一个面向 `SSA vs others` 的 WSI Agent 原型搭起来了，整体逻辑参考了 CPathAgent 的三阶段设计。当前系统会先在低倍 overview 上筛区域，再生成多倍率导航轨迹，最后输出逐步观察日志和英文病理报告。现阶段已经可以在真实切片上跑通，并产出可回放、可审核的结果。下一步重点不是再堆功能，而是把更强的多模态 backend 接进来，在一个小规模 pilot 子集上做系统性评测和人工复核，把“流程跑通”推进到“结果更可信”。"

# 00. 项目结构与数据流

## 模块目标

本项目围绕腺瘤相关 WSI 做 slide-level 计算病理建模，主线是：

```text
WSI
  -> CLAM coords h5
  -> UNI features h5 / pt
  -> ready csv / split
  -> slide-level models
  -> metrics / heatmaps / reports
```

## 关键目录

| 路径 | 作用 |
| --- | --- |
| `/data15/data15_5/yuexin2/adenoma` | 项目根目录 |
| `config/` | 各倍率预处理、特征提取、训练配置 |
| `scripts/` | 数据准备、训练、评估、热图、Route C 等入口脚本 |
| `data/` | 标签、ready manifest、fold split、统计文件 |
| `runs/` | CLAM 预处理输出 |
| `outputs/` | 特征、训练结果、分析图表与报告 |
| `transmil/` | 项目内自实现 TransMIL |
| `patch_gcn/` | PatchGCN 数据、图构建、模型与训练 |
| `dsmil/` | DSMIL 数据、模型与训练 |
| `transmil_official/` | official-style TransMIL 路线 |
| `/data15/data15_5/yuexin2/MIST` | MIST 代码与本地 manifest；hp+yx 多分类 manifest 也同步到外置数据盘 |

外部依赖：

| 路径 | 作用 |
| --- | --- |
| `/data15/data15_5/yuexin2/CLAM` | CLAM 预处理、MIL/CLAM 训练基础代码 |
| `/data15/data15_5/yuexin2/MAG-GLTrans` | MAG-GLTrans 相关路线保留 |
| `/data15/data15_5/yuexin2/Patho-R1` | Route C / Patho-R1 相关路线保留 |

## 当前正式结果口径

- 数据：`Adenoma_yx`
- 特征：`UNI`
- 划分：`fold-5`
- 主任务：`SSL vs others`
- 模型：`CLAM-SB`、`TransMIL`、`PatchGCN`、`DSMIL`
- 扩展：`dysplasia vs no_dysplasia` 二阶段任务作为探索性结果
- 多分类扩展：MIST hp+yx 使用 UNI low4cat 特征，当前实际运行的 manifest 为 `11class`，训练参数 `num_classes=11`；如果采用最新十分类口径，则把 `ssl with highgrade dysplasia` 合并回 `ssl`，其余高等级 `TSA/TA/TVA` 单独保留。

旧 `fold-0` 结果和图表保留为历史探索参考，不作为当前正式主口径。

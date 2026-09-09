## 当前主线：20X 完整训练 + 2.5X 低倍训练

当前正式主线已收敛为：

- 任务：`SSL vs Others`
- 标签定义：
  - `SSL = Sessile serrated adenoma`
  - `others = 其余全部类型`
- 任务：`SSL vs Others`
- 高倍路线：`CLAM MIL @20X`
- 低倍路线：`CLAM MIL @2.5X`
- 划分：`slide-level`
- 范围：`fold-0` 单折闭环

当前主线不是直接训练 `MAG-GLTrans`，而是：

1. 完成 `CLAM` 预处理
2. 完成 `20X` 特征提取
3. 跑 `CLAM MIL @20X`
4. 完成 `2.5X` patch 与特征提取
5. 跑 `CLAM MIL @2.5X`
6. 后续再恢复 Patho-R1 与进入 `MAG-GLTrans`

### Patho-R1 路线（保留，暂不使用）

Patho-R1 相关脚本与结果保留，但当前不作为正式主线。

优点：

- 利用 AI 推理能力，自动识别感兴趣区域
- 减少手动阈值调参，适应复杂组织
- 结合低分辨率推理与高分辨率提取

缺点：

- 需要 Patho-R1 环境和模型
- 推理时间较长
- 选区准确性依赖模型性能

#### Phase 1C：生成缩略图

目标：

- 为每张 WSI 生成 1X 缩略图

使用脚本：

- 基于 openslide 生成缩略图

输出：

- `outputs/adenoma_yx_thumbnails/1X/*.png`

代码示例：

```python
import openslide
from PIL import Image

slide = openslide.OpenSlide('/data15/zhengke_usb/Adenoma_yx/slide.svs')
thumbnail = slide.get_thumbnail((slide.dimensions[0] // slide.level_downsamples[0], slide.dimensions[1] // slide.level_downsamples[0]))
thumbnail.save('thumbnail.png')
```

#### Phase 2C：使用 Patho-R1 进行选区

目标：

- 在 1X 缩略图上运行 Patho-R1 推理，识别组织区域

使用模型：

- Patho-R1-7B 或 Patho-R1-3B
- 当前项目也提供一个 `heuristic` 模式，作为没有模型权限时的 smoke test 兜底

推理提示：

- "Identify and outline the regions containing adenoma tissue in this pathology image thumbnail."

输出：

- 缩略图上的 bounding boxes
- 结构化 `json`
- 可视化框图

代码示例：

```bash
/data15/data15_5/yuexin2/anaconda3/envs/patho-r1/bin/python \
  /data15/data15_5/yuexin2/adenoma/scripts/patho_r1_route_c_select.py \
  --slide-path /data15/zhengke_usb/Adenoma_yx/slide.svs \
  --output-dir /data15/data15_5/yuexin2/adenoma/outputs/route_c_patho_r1/slide_id \
  --mode patho-r1 \
  --model-id WenchuanZhang/Patho-R1-3B \
  --hf-token <YOUR_HF_TOKEN>
```

当前项目中仍保留的脚本：

- 选区脚本：
  - `/data15/data15_5/yuexin2/adenoma/scripts/patho_r1_route_c_select.py`
- 批处理脚本：
  - `/data15/data15_5/yuexin2/adenoma/scripts/run_route_c_batch.py`
- smoke 脚本：
  - `/data15/data15_5/yuexin2/adenoma/scripts/run_route_c_smoke.sh`

输出：

- 缩略图：`*_thumbnail.png`
- 原始响应：`*_route_c_raw_response.txt`
- 结构化框：`*_route_c_boxes.json`
- 选区可视化：`*_route_c_boxes.png`

#### Phase 3C：转换选区为 patch 坐标

目标：

- 将 Patho-R1 的选区转换为 CLAM 可用的 patch 坐标

方法：

- 解析 Patho-R1 输出中的 bounding box
- 将 thumbnail 坐标缩放到 level 0
- 以固定 patch size / step size 生成网格坐标
- 保存成与 CLAM 兼容的 `coords .h5`

当前项目脚本：

- `/data15/data15_5/yuexin2/adenoma/scripts/route_c_boxes_to_h5.py`

示例：

```bash
/data15/data15_5/yuexin2/anaconda3/envs/patho-r1/bin/python \
  /data15/data15_5/yuexin2/adenoma/scripts/route_c_boxes_to_h5.py \
  --boxes-json /data15/data15_5/yuexin2/adenoma/outputs/route_c_patho_r1/slide_id/slide_id_route_c_boxes.json \
  --output-h5 /data15/data15_5/yuexin2/adenoma/outputs/route_c_coords/slide_id.h5 \
  --patch-size 256 \
  --step-size 256 \
  --patch-level 0
```

暂停原因：

- CPU 上真实 `Patho-R1-3B` 推理过慢
- 会拖住整批处理，不适合作为当前主流程

重新启用条件：

- 换到 GPU 机器
- 或者后续有更轻量的 VLM / 更稳的推理部署方式

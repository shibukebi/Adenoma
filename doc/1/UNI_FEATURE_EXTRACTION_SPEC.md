# UNI 特征提取物理尺度说明

这份文档描述 adenoma 项目中 UNI 特征提取的坐标、物理视野和输入标准化约定。

核心原则：

- 坐标一律使用 WSI level 0 坐标系
- 特征提取不应依赖 `level=2` 这类金字塔层索引
- 物理视野大小用 `physical_level_0_extent` 表示
- 模型输入大小用 `target_patch_size` 表示

## 1. 坐标约定

patch h5 中的 `coords` 表示 patch 左上角在 level 0 中的位置：

```text
coord = [level0_x, level0_y]
```

计算坐标时严禁使用：

```text
patch_level
source_level_downsample
level index
```

这些字段只能作为 legacy provenance，用于解释旧数据是如何生成的。

## 2. 物理视野字段

UNI 特征提取输出 h5 会在 `features.attrs` 和 `coords.attrs` 写入：

```text
physical_level_0_extent
```

含义：

```text
每个 patch 在 level 0 上覆盖的边长，单位是 level0 pixels
```

例如：

```text
physical_level_0_extent = 4096
```

表示该 patch 覆盖：

```text
4096 x 4096 level0 pixels
```

消费 UNI 特征时应优先读取这个字段，而不是从 `patch_level * patch_size` 或文件名反推。

## 3. Legacy h5 字段

旧的 CLAM patch h5 仍可能包含：

```text
patch_level
patch_size
downsample
```

这些字段只能用于计算旧数据的实际物理覆盖范围：

```text
legacy_physical_level_0_extent = patch_size * downsample
```

例如：

```text
patch_size = 256
downsample = 8
legacy_physical_level_0_extent = 2048
```

或者：

```text
patch_size = 256
downsample = 16
legacy_physical_level_0_extent = 4096
```

由于不同 WSI 的同一 `patch_level` 可能对应不同 downsample，因此不能把 `patch_level = 2` 理解成固定物理尺度。

## 4. 标准化读取逻辑

`extract_features_route_a.py` 支持：

```text
--expected-physical-level0-extent 4096
```

启用后，特征提取不再使用 h5 中的 `patch_level` 读取图像，而是：

1. 读取 h5 中的 level 0 坐标
2. 从 WSI level 0 读取 `4096 x 4096` 的物理视野
3. 交给 UNI transform resize 到 `target_patch_size`
4. 提取 UNI feature

因此，即使旧 h5 的实际范围只有 `2048 x 2048`，进入 UNI 前也会被统一为期望的 level 0 物理视野。

## 5. 模型输入大小

`target_patch_size` 是送入 UNI 前的像素尺寸，当前配置为：

```text
FEATURE_TARGET_PATCH_SIZE = 224
```

它不代表 WSI 物理视野，只代表模型输入图像尺寸。

标准流程是：

```text
level0 physical crop -> resize to 224 x 224 -> UNI encoder
```

## 6. 5x 当前配置

当前 yx 5x UNI 配置：

```text
FEATURE_EXPECTED_PHYSICAL_LEVEL0_EXTENT = 4096
FEATURE_TARGET_PATCH_SIZE = 224
```

也就是说，5x UNI 特征统一按：

```text
4096 x 4096 level0 pixels
```

作为物理视野，再 resize 成：

```text
224 x 224
```

送入 UNI。

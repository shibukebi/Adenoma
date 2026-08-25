# Pathology Evidence Annotation System（MVP）

面向两阶段数字病理 evidence study 的 React + TypeScript 前端演示：A 阶段专家主动标记 1–5 个 ROI；B 阶段将去重后的候选以固定盲 ID / 固定随机顺序交给裁决者。所有演示数据通过浏览器 `localStorage` 持久化。

## 启动

要求 Node.js 16（当前依赖固定为 React 18 / Vite 4，兼容 Node 16）。

```bash
cd annotation-system
npm install
npm run dev
```

测试与生产构建：

```bash
npm test
npm run build
```

首次进入可用顶栏“演示身份”切换病理医师、盲法裁决者、管理员。若希望还原 demo，清理浏览器中键 `pathology-evidence-annotation-mvp-v1`。

## 关键实验边界

- `src/types.ts`：geometry（level-0 逻辑像素）、physical FOV、evidence、round、revision、audit 与仅管理员可见的 `sourceType` / `sourceRunId` 数据模型。
- `src/store.ts`：`blindCandidateAdapter` 使用显式 allowlist 构建裁决 API DTO（只有 blind ID、slide/geometry 与裁决字段），不会通过未来新增字段泄露来源；不依赖仅在 UI 隐藏来源。`cloneInitialState` / `createLocalId` 不使用 `structuredClone` 或 `crypto.randomUUID`，兼容 Firefox 86。
- 初始轮支持 1–5 个 ROI 与草稿/提交；管理员 Freeze 后策略层禁止编辑。真实后端应继续在 API 与数据库执行同样的权限判断。
- B 阶段不能新建 ROI；微调时 `refineCandidate` 把候选初始 geometry 保存到 `originalGeometry` 并增加 revision。
- `unionMembers` 显式保存 `union candidate → proposal/member` 关系及 source/run/rank/model version。管理员导出按 candidate 聚合保留 members/provenance，供最终解盲统计；盲法 DTO 永不包含它。

`public/demo-slide.jpg` 是项目内复制的真实 H&E 缩略图（demo `CASE-017` 的 mock 元数据使用原始 SVS 的 level-0 尺寸 23904 × 22498 与 mpp_x/y ≈ 0.2517 µm/px），仅作为模拟 WSI 背景，不跨目录运行时读取。所有 ROI overlay 与拖拽都转换到 level-0 坐标；当前 viewer 仍只模拟平移/缩放、视野和矩形 ROI，没有声称能替代真正 WSI tile viewer。

## 生产接入边界

前端中 `SlideViewer` 应替换为 OpenSeadragon。后端以 FastAPI 提供受认证的 Deep Zoom / IIIF manifest 和 tile endpoints；OpenSlide 保持 WSI 只读并按所需 level 输出 tiles。ROI 交互应由 Annotorious 的 OpenSeadragon adapter 处理，并将结果规范化为 level-0 pixel coordinates / optional polygon。

FastAPI 应分别提供：初始标注 API、**不含 provenance 的** adjudicator candidate API、仅管理员可访问的 union / export API。PostgreSQL 表建议由 `cases`、`slides`、`annotation_rounds`、`roi_proposals`、`union_candidates`、`union_candidate_members`、`expert_annotations`、`evidence_tags`、`audit_log` 组成。去重、固定随机 seed、冻结/修订、RLS/API schema 权限判断必须置于服务端；本 MVP 的浏览器数据仅用于流程演示。

## Tests

`tests/store.test.ts` 覆盖：allowlist DTO 不泄露未知 provenance 字段、level-0 geometry/反向拖拽映射、geometry revision 留存原始值并递增版本、双阶段冻结策略、RFC4180 nested CSV，以及管理员导出的 Union member provenance。

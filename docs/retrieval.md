# 全局检索与粗定位 `init` — query 的初始位姿是怎么来的

> 背景：重定位 [`refine_query`](../odo/reloc_pnp.py#L101) 用 triangulation + PnP **精修**出 query 的位姿，
> 但它需要一个**粗略初值 `init`** 当起点。本文件讲 `init` 是怎么得到的——答案是**全局图像检索**
> （[`odo/retrieval.py`](../odo/retrieval.py)），不涉及任何三角化/PnP。

命令：
```bash
python -m odo.retrieval --desc outputs/10_vpr/desc --model megaloc --cams fl fr r \
       --out outputs/11_retrieval_megaloc --qsplit tune
```

---

## 0. 一句话

`init` = 拿 query 图的**全局描述子**（MegaLoc）在数据库里找**最像的 k 张老照片**，取 top1（或 top-k 相似度加权）**那些老照片当时的 GT 位姿**，当作 query 的粗略初始位姿。直觉：“当前这帧最像第 723 号老照片，那张拍摄时车在赛道 (x,y,yaw)，所以当前车大概也在那附近。” 精度只到“赛道这一段”的量级，交给后续 PnP 精修。

---

## 1. 核心思路：借「已知位置的老照片」给当前帧定位

当前 query 帧不知道车在哪，但训练库里有大量**拍摄时 GT 位置已知**的老照片。只要找到和当前帧“拍的是同一片场景”的老照片，就能借它的已知位置粗估当前位置——像看到陌生街景照却认出背景里那栋楼，就知道拍照人大概站哪。

全局描述子（MegaLoc，一种 VPR / 地点识别模型）把每张图压成一个向量，“看起来像同一个地方”的图向量彼此接近。于是“找最像的老照片”= 向量余弦相似度最高。

---

## 2. 三步得到 `init`

### 2.1 对齐多相机描述子到同一时间线（[`aligned`](../odo/retrieval.py#L22)）
每个 query 时刻有多路相机。以参考相机（`fl`）的时间线为准，其它相机按**最近 stamp** 对齐，取各自描述子。这样一个 query 时刻 = 每个相机各一个全局描述子。

### 2.2 query↔数据库相似度（[`similarity`](../odo/retrieval.py#L35)）
```python
S = Σ_cam ( Dq_cam @ Ddb_cam.T ) / n_cam     # 各相机余弦相似度，按相机数平均
```
`S[i, j]` = 第 i 个 query 与第 j 张数据库老照片的平均相似度（多相机投票）。

### 2.3 相似度 → 位姿（[`estimate`](../odo/retrieval.py#L42)）
```python
order = argsort(-S)[:, :k]      # 每个 query 取相似度最高的 k 张老照片（默认 k=5）
best  = order[:, 0]             # 最像的那张
```
- **`mode='top1'`**（[L46](../odo/retrieval.py#L46)）：直接用最像那张老照片的 GT 位姿。
- **`mode='topk'`（默认）**（[L48-L55](../odo/retrieval.py#L48)）：在 `best` 附近 `cluster_m`（默认 10 m）内的几张候选里，按相似度 softmax 加权平均它们的 GT 位姿：
  ```python
  near = |xy_db − xy_best| < cluster_m          # 只聚合空间上靠近 best 的候选（剔除“撞脸”的远处地点）
  w    = softmax(S[near] / temp)                # temp=0.02，越像权重越大
  ex, ey = Σ w·x_db, Σ w·y_db                   # 加权平均位置
  eyaw   = angle(Σ w·e^{i·yaw_db})              # 圆周量加权平均朝向
  ```
  `cluster_m` 门控很重要：不同地点也可能长得像（感知混叠），只聚合离 best 近的候选，避免把两处远地点的位姿平均到一起。

> 位姿的“真值”来自老照片：`db = data.interp_gt(tdb)`（[`retrieval.run` L82](../odo/retrieval.py#L82)），即各数据库帧的 GT (x, y, yaw)。

### 2.4 落盘（[`run`](../odo/retrieval.py#L78)）
```python
x, y, yaw, smax = estimate(S, db, ...)                       # L84
pd.DataFrame(dict(timestamp_ns=tq, x=x, y=y, yaw=yaw)).to_csv('pred.csv')   # L85
np.savez('retrieval.npz', tq=tq, best=argmax(S), top=argsort(-S)[:, :50], tdb=tdb)   # L94
```
- `pred.csv`：每个 query 一行 `(timestamp_ns, x, y, yaw)` = 粗位姿。
- `retrieval.npz`：`tq`（query 时间戳）、`best`/`top`（候选数据库帧索引）、`tdb`（数据库帧时间戳）——供 `reloc_pnp` 挑候选用。

---

## 3. `reloc_pnp` 如何消费 `init`

[`reloc_pnp.main`](../odo/reloc_pnp.py#L163) 读 `pred.csv` + `retrieval.npz`：
```python
init = (pred.x[i], pred.y[i], pred.yaw[i])          # L175: 直接来自 retrieval 的 pred.csv
```
再做**几何重排序**，试多个空间上分散（≥8 m）的检索候选（[L176-L187](../odo/reloc_pnp.py#L176)）：
```python
init_c = init if ci==0 else (dbxy[j][0], dbxy[j][1], interp_gt(tdb[j]).yaw)   # L184
p, info = refine_query(ext, gt6, tq[i], init_c, tdb[j], cams, **kw)
if info['status'] == 'ok': break                     # 第一个 PnP 精修成功的候选即采纳
```
- 第一候选用 `pred.csv` 的 `init`；后续候选直接用**那张候选老照片的 GT 位姿**当起点。
- 若所有候选都精修失败（`sel<6` / `n_corr<30` / `jump`，见 [map_build.md 附录](map_build.md)），`refine_query` 原样退回 `init` —— **这就是“回退初值”**：放弃几何精修，拿检索粗位姿交差，保证每帧都有输出（coverage 100%，见 [`online.py:13-15`](../odo/online.py#L13)）。

---

## 4. query / 数据库划分（`qsplit`）

见 [`query_db`](../odo/retrieval.py#L63)：
- `qsplit='val'`：query = 验证圈，db = 整个 train。
- `qsplit='tune'`（dt 标定用）：query = tune lap **L5**，db = train **去掉 L5 ±10 s**——保证 query 不会检索到自己或紧邻帧（跨圈重定位，不是自我对照）。详见 [map_build.md 附录 A.2](map_build.md)。

---

## 5. 串起来

```
query 图(多相机)
  → aligned: 各相机描述子对齐到参考时间线
  → similarity: 与数据库老照片的全局描述子算余弦相似度 S（多相机平均）
  → estimate: 取 top-k、cluster_m 内按相似度加权 → 借老照片 GT 位姿得 (x,y,yaw)
  → pred.csv (init, 粗位姿) + retrieval.npz (候选帧)
        │
  reloc_pnp: 以 init 为起点，对多个候选做 triangulation + PnP 精修
        └─ 精修成功 → 精位姿;  全失败 → 回退 init(粗位姿)
```

**一句话：`init` 是纯图像检索的产物——全局描述子找最像的老照片，借其已知 GT 位姿得到的粗定位；它只负责给 PnP 一个起点，自身不含任何三角化/PnP 几何。**

---

## 相关文件

| 文件 | 作用 |
|---|---|
| [`odo/retrieval.py`](../odo/retrieval.py) | 全局检索 → 粗位姿 `pred.csv`（`aligned`/`similarity`/`estimate`/`query_db`/`run`）|
| [`odo/vpr.py`](../odo/vpr.py) | VPR 全局描述子模型（MegaLoc）加载/提取 |
| [`odo/reloc_pnp.py`](../odo/reloc_pnp.py) | 读 `init` 做 triangulation + PnP 精修；失败回退 `init` |
| [`odo/data.py`](../odo/data.py) | `interp_gt`（老照片 GT 位姿）、`SPLITS`/`TUNE_LAP`、`CAMS` |
| [`docs/map_build.md`](map_build.md) | 建图 + `DT_NS` 标定（含 query/db 划分、sel/n_corr 门槛）|

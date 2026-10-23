# 重定位（Relocalise）：从丢失中跳回正确位置

> Brief 的那一句：
> *"Relocalise. After 1 s without an accepted fix, MegaLoc retrieval proposes 4 spatially distinct candidates, each
> verified by wide-radius matching and PnP. A jump is accepted only if it is plausible for the time without a fix
> (0.5 m + 0.3 m/s, 0.3° + 0.1°/s) or has at least 300 inliers."*
>
> 讲的是在线定位 [`odo/online.py`](../odo/online.py) 的**丢失恢复**路径。核心：常规定位靠当前状态 `X` 在地图里查近邻，
> 丢失时 `X` 不可信 → 退回 **MegaLoc 全局检索**猜候选，再验证、谨慎接受跳变。

---

## 0. 常规定位 vs 重定位（最关键的区别）

两条路最后都汇到同一个 [`measure`](../odo/online.py#L504)（选地图帧 → 引导匹配 → PnP），**区别只在「起点 prior 从哪来、跑几次」**。`measure` 内部选地图帧的逻辑（`Map.select`，每相机最多 `n_pass=3` 帧）**两者完全相同**：

| | 常规定位（每帧） | 重定位（丢失时） |
|---|---|---|
| 前提 | EKF 的 `X` 大致可信（motion 在跟踪） | `X` 不可信（跟丢了） |
| 起点 prior | **1 个**：EKF 当前状态 `X` | **5 个**：`X` + MegaLoc 检索的 4 个候选 prior 位姿（[`Reloc.candidates`](../odo/online.py#L469)）|
| prior 怎么来 | 直接用 `X` | 全局检索：当前帧 vs 地图帧全局相似度 → 取最像且 ≥8 m 分散的地图帧的 **GT 位姿** |
| 跑几次 `measure` | 1 次 | 5 次（每个 prior 一次），取 PnP **内点最多**的为 best |
| 选地图帧（`measure` 内） | `Map.select`：`X` 查 **KD-tree** 近邻，每相机 ≤`n_pass=3` 帧 | **同左**（prior 换成各候选，逻辑一样） |
| 引导匹配半径 `r1` | 小，`r1≈24`（prior 准，窗口小更精更抗错配） | 大，`reloc_r1=60`（prior 可能偏，窗口大才兜住真匹配）|
| 代价 / 频率 | 便宜（1 次 measure）/ **每帧** | 贵（全库描述子比对 + 5 次 measure）/ 仅丢失且 ≥1 s 后、隔 0.5 s |
| 接受门槛 | 马氏门控 χ²=16（[ekf_update.md](ekf_update.md)） | 门控 **+** 跳变可信性（随丢失时长放宽）或 ≥300 内点 |
| 结果 | 卡尔曼更新（渐进拉向 fix） | reset（状态直接**跳**到候选） |

> 一句话：常规定位是“我大概知道在哪，确认一下”；重定位是“我不知道在哪了，全局找回来”。**“4 个候选”指 4 个起点 prior 位姿，不是“用 4 个地图帧定位”**——每个 prior 各自进 `measure`，内部照样取最多 3 个地图帧。全局检索（MegaLoc）不是每帧都做，只在跟丢时兜底。

---

## 1. 触发：≥1 s 没有被接受的 fix（[L703-L704](../odo/online.py#L703)）

```python
if reloc and since > reloc_after(1.0) and (stamp - last_reloc)/1e9 >= reloc_every(0.5) and \
   not nomap and (p_meas is None or n_inl < reloc_min_inl(80) or n_rej > 0):
```
- `since > reloc_after(1.0)`：**距上次被接受 fix 已超过 1 s**（`since=(stamp-last_fix)/1e9`，`last_fix` 只在 fix/reset 时更新）——即 brief 的 "1 s without an accepted fix"。
- `reloc_every=0.5`：两次重定位至少隔 0.5 s，避免过频。
- 且常规定位救不回：`p_meas is None`（PnP 失败）或内点 `< reloc_min_inl(80)` 或刚有拒绝 `n_rej>0`。
- 参数见 [`DEFAULT` L589-590](../odo/online.py#L589)。

---

## 2. 候选生成：MegaLoc → 4 个候选 **prior 位姿**（[`Reloc.candidates`](../odo/online.py#L469)）

```python
cands = reloc.candidates(当前帧各相机图, prior=X)    # L708
```
内部（[L469-L500](../odo/online.py#L469)）：
1. **全局相似度比对**：用 **MegaLoc 全局描述子**（整幅图压成一个向量，[L485](../odo/online.py#L485)）把当前帧（多相机平均余弦相似度 `S = Σ self.D @ d`，[L486](../odo/online.py#L486)）与**地图库**（训练图的全局描述子 `self.D`，由 [`odo.vpr`](../odo/vpr.py) 预先算好）比，取最像的 top-50 地图帧（[L488](../odo/online.py#L488)）。
2. **挑空间分散的 4 个**：从最像往下取，要求彼此 `min_sep=8 m`（[L496](../odo/online.py#L496)），凑够 `n_cand=4`（[L469](../odo/online.py#L469)）。
3. **返回这些地图帧拍摄时的 GT 位姿**（[L500](../odo/online.py#L500) `self.pose[j]`）作为候选 **prior**。

> **关键：返回的是“候选 prior 位姿”，不是“定位用的地图帧”。** 全局检索只回答“当前帧最像哪几个地图帧 → 我大概在那些地图帧的位置”，给出**位置猜测起点**；精确位姿留给第 3 步的 PnP。
>
> **全局 vs 局部**：这一步用 **MegaLoc 全局描述子**比“整幅图像像不像同一个地方”（只定位到“哪个地图帧附近”，粗）；第 3 步才用 **SuperPoint 局部特征 + 3D 点 + 几何**精确求位姿（细）。两者分工同离线 `init`（见 [retrieval.md](retrieval.md)）。
>
> “spatially distinct”（≥8 m）防感知混叠：赛道不同处可能长得像，候选 prior 分散能覆盖多个假设，不至于全押错一处。注意这个 8 m 说的是**候选 prior 位姿**彼此分散，与下一步每次 `measure` 内 `Map.select` 选地图帧（按“趟”`pass_gap_ns=5 s` 分组）是**两个不同层级**。

---

## 3. 验证：逐候选 prior 跑 `measure`（大半径 PnP），取 best（[L711-L717](../odo/online.py#L711)）

对 `[X] + cands`（当前状态 + 4 候选 prior，共 5 个**起点**）各跑一次完整 `measure`，取内点最多者：
```python
for prior in [X] + cands:
    pr, cr, ir = measure(..., prior, cfg['reloc_r1'])    # reloc_r1=60：大搜索半径
    if pr and (best is None or ir.n_inl > best.n_inl):
        best = (pr, cr, ir)
if best and best.n_inl >= reloc_min_inl(80):
    p_meas, cov_meas, info = best; relocated = True
```
- **“4 个候选”是 4 个起点 prior，不是“用 4 个地图帧定位”**。每个 prior **各自**进 `measure`，内部照样用 [`Map.select`](../odo/online.py#L77) 每相机取**最多 `n_pass=3` 个地图帧**（和常规定位**完全一样**）。
  - 常规：1 个 prior（`X`）→ 跑 1 次 `measure`。
  - 重定位：5 个 prior → 跑 5 次 `measure`，**取 PnP 内点最多的那次结果**为 `best`。
- `measure` 同常规路径，只是搜索半径放大到 `reloc_r1=60`（平时 `r1≈24`）——"**wide-radius**"：prior 来自检索可能偏，需大窗口才兜得住真匹配（半径如何作用见 [pnp.md](pnp.md)）。
- 为什么跑多个取 best：丢失时检索可能给出几个都像的地点（感知混叠），单押一个可能押错；让每个候选各跑一遍，**PnP 内点数当证据**——真正对的地点内点明显最多。
- `best` 内点 `≥ reloc_min_inl(80)` 才当作本帧测量 `p_meas` 并标 `relocated=True`。

---

## 4. 跳变接受：幅度可信 或 ≥300 内点（[L743-L752](../odo/online.py#L743)）

重定位会让状态**跳**到候选，跳错代价大（直接漂十几米），所以在普通门控基础上再加一道**随丢失时长放宽的宽松门控**：
```python
sig     = reloc_sig0(0.5) + reloc_sig_rate(0.3) * since         # 位置裕度 = 0.5 m + 0.3 m/s × 丢失时长
sig_yaw = radians(reloc_yaw0_deg(0.3) + reloc_yaw_rate_deg(0.1) * since)  # 朝向裕度 = 0.3° + 0.1°/s × 时长
d2_loose = nu @ (S + diag(sig², sig², sig_yaw²))⁻¹ @ nu
if relocated and d2 >= gate and d2_loose >= gate and n_inl < reloc_strong_inl(300):
    status = 'reloc_rejected'          # 跳变不可信 → 拒绝（L748-L749）
elif relocated and d2 >= gate:
    X = p_meas; P = reset              # 接受 → reset 到候选（L750-L752）
```
- **`0.5 m + 0.3 m/s`**：允许跳变距离随丢失时间线性放宽。刚丢（1 s）只允许 ~0.8 m；丢得越久、航位漂得越多，允许跳得越远。注释（[L742](../odo/online.py#L742)）：航位漂 ~0.06 m/s，用 0.3 m/s 留 5 倍裕度。
- **`0.3° + 0.1°/s`**：朝向跳变裕度。注释（[L745-L746](../odo/online.py#L745)）：yaw 航位漂极慢（1 s ~0.09°），所以大 yaw 跳变基本不可信（Exp24：2.5° 错跳 → 漂 7 m）。
- **"或 ≥300 内点"**（`reloc_strong_inl=300`，[L590](../odo/online.py#L590)）：若候选 PnP 内点特别多（证据压倒），**即便超出裕度也接受**——300 内点几乎不可能是错配。
- 逻辑：`relocated` 且普通门控 `d2` 与放宽门控 `d2_loose` 都没过、且内点 `<300` → **拒绝**；否则 → **接受，reset 到候选**。

> **跳变要么"幅度合理"（随时间放宽的裕度内），要么"证据压倒"（≥300 内点），否则宁可不跳。**

---

## 5. 整体流程

```
每帧:
  常规定位 p_meas = measure(prior=X, r1≈24)             # 1 个 prior；内部 Map.select 每相机≤3帧
  若 (since>1s 且 常规内点<80/失败/有拒绝):             # 判定“丢失” (L703-704)
    cands = MegaLoc 全局相似度 → 最像且 ≥8m 分散的 4 个地图帧的 GT 位姿  # 候选 prior (L708)
    best  = 对 [X]+cands 这 5 个 prior 各跑 measure(reloc_r1=60)，取 PnP 内点最多  # (L711-714)
            （每个 prior 内部仍 Map.select 每相机≤3帧——与常规相同）
    若 best 内点≥80 → p_meas=best, relocated=True         # (L714-717)
  更新 (L743-752):
    跳变在 (0.5m+0.3m/s, 0.3°+0.1°/s) 内 或 内点≥300 → 接受, reset (status='reloc')
    否则 → 拒绝 (status='reloc_rejected')
```

**一句话：距上次被接受 fix >1 s 且常规定位救不回（内点<80 或失败）时触发重定位；用 MegaLoc 全局描述子把当前帧和地图帧比相似度，取最像且彼此 ≥8 m 分散的 4 个地图帧的 GT 位姿作为候选 prior（全局检索只定“在哪个地图帧附近”，粗）；对「当前状态 + 4 候选」这 5 个起点 prior 各跑一次 `measure`（内部仍 `Map.select` 每相机≤3帧、大半径 60 m 引导匹配 + PnP 精定位），取 PnP 内点最多者；该跳变仅在幅度可信（随丢失时长放宽的 0.5 m+0.3 m/s、0.3°+0.1°/s）或内点≥300 时才被接受并 reset。与常规的本质差别只在「起点 prior 从哪来、跑几次」：常规 1 个 prior（`X`），重定位 5 个 prior（检索候选）；`measure` 内选地图帧的逻辑两者完全相同。**

---

## 相关文件

| 文件 | 作用 |
|---|---|
| [`odo/online.py`](../odo/online.py) | `Reloc`（全局检索候选）、`Map.select`（KD-tree 近邻）、`measure`、更新步（跳变接受）|
| [`odo/vpr.py`](../odo/vpr.py) | MegaLoc 全局描述子（预算地图库描述子）|
| [`docs/ekf_update.md`](ekf_update.md) | 门控 / 弱 fix / challenger（重定位 reset 与 challenger reset 对照）|
| [`docs/pnp.md`](pnp.md) | `measure` 内部的匹配 + PnP 几何 |
| [`docs/retrieval.md`](retrieval.md) | 全局检索给粗位姿的同类机制（离线 `qsplit` 版）|

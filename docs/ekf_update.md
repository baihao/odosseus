# EKF 更新步：马氏门控 / 弱 fix / 挑战者

> Brief 的那一句：
> *"Update. A Mahalanobis gate (χ² 16) accepts or rejects the fix. Weak fixes (fewer than 60 inliers) update the
> position only. If 8 rejected fixes agree with each other, they replace the state (challenger), so the filter can
> escape a wrong state."*
>
> 讲的是在线定位 [`odo/online.py`](../odo/online.py) 的 **EKF 更新步**：拿到一次视觉 PnP fix 后，决定
> **信不信、信多少、万一卡在错状态怎么逃出来**。三件事：门控、弱 fix 只更位置、挑战者重置。

---

## 0. 三个关键量（更新前已算好）

主循环里先做视觉测量再算新息（[L701](../odo/online.py#L701)、[L722](../odo/online.py#L722)、[L732-L734](../odo/online.py#L732)）：
```python
p_meas, cov_meas, info = measure(...)        # 视觉 PnP 观测：位姿 [x,y,yaw] + 协方差 + n_inl
Rm = cov_meas * 4 + floor                     # 测量噪声 R（见 online_ekf.md）
nu = p_meas - X                               # 新息 = 观测 − 当前状态（预测）
S  = P + Rm                                    # 新息协方差 = 预测不确定度 + 测量噪声
d2 = nu @ S⁻¹ @ nu                             # 马氏距离²
```
三个主角别混（下面反复用）：

| 变量 | 是什么 | 来源 |
|---|---|---|
| `X` | EKF **当前状态**（现任，滤波相信的位姿） | motion 预测 + 历史 fix |
| `p_meas` | **这一帧视觉观测** fix `(x,y,yaw)` | [`measure`](../odo/online.py#L504) → `solve_pnp`（PnP，见 [pnp.md](pnp.md)）|
| `chal` | **挑战者**（之前某次被拒 `p_meas` 的快照） | `chal = p_meas.copy()` |

---

## 1. 马氏门控 χ²=16 —— 接受 / 拒绝这次 fix

`d2` 不是看 fix 偏离预测几米，而是**用协方差 S 归一化后**偏了几个“标准差”。阈值 [`gate=16.0`（DEFAULT L588）](../odo/online.py#L588)，对 3 自由度约“3-sigma”置信。

- **`d2 < 16` → 接受**，做标准卡尔曼更新（[L753-L755](../odo/online.py#L753)）：
  ```python
  Kg = P @ inv(S)
  X  = X + Kg @ nu;  P = (I - Kg) @ P      # 把状态拉向 fix，更新不确定度
  status = 'fix'; n_rej = 0; chal = None   # 清零拒绝计数、清空挑战者
  ```
- **`d2 ≥ 16` → 拒绝**，不更新，`status='gated'`，拒绝计数 `n_rej++`（[L758-L759](../odo/online.py#L758)），再进挑战者逻辑（§3）。

> 为什么用马氏距离而非欧氏：预测很不确定（P 大）时允许 fix 偏远些；预测笃定（P 小）时一点偏差就该警惕。门控挡掉明显错配 / 错误重定位。

---

## 2. 弱 fix（内点 < 60）只更新位置，不动 yaw

这件事在**构造 `Rm` 时**就做了（[L728-L731](../odo/online.py#L728)，`yaw_min_inl=60`）：
```python
if info['n_inl'] < cfg['yaw_min_inl']:        # 内点不足 60
    Rm[2,:2] = Rm[:2,2] = 0; Rm[2,2] = radians(30)²   # 把 yaw 测量方差设成巨大
```
yaw 方向测量噪声设爆 → 卡尔曼增益在 yaw 上≈0 → 这次 fix **只修 x,y，不修朝向**。

原因（注释 [L729](../odo/online.py#L729) 引 Exp24）：少量内点解出的 PnP yaw 不可靠，不如车端航向率（`wz`）推算准。实测一个 30 内点的 fix 把 yaw 带偏 1.6° → 滤波随后漂 11 m。所以**弱 fix 的位置可用、朝向不可信，就只采纳位置**。

---

## 3. 挑战者（challenger）—— 8 个被拒 fix 互相印证 → 夺权

**逃离错误状态的兜底机制。挑战者是“观测”（视觉 fix），不是 motion 预测。**

### 为什么需要
设想 `X` 卡在错位置（如之前一次错误重定位）。此后每次视觉 fix 其实指向**真实位置**，但真实位置离错的 `X` 很远 → `d2 ≥ gate` → **fix 全被拒**，滤波继续用错 `X` 空转。难点：单看“fix 偏离 X 很远”无法区分是 **fix 错**（偶发错配，该拒）还是 **X 错**（滤波跑飞，该信 fix）。

### 怎么判：观测 vs 观测（不是和 X 比）
关键——判断被拒 fix 是否可信，**不是和当前 X 比**（它们本来就差很远才被拒），而是把**被拒的 fix 们彼此比**。代码（[L758-L772](../odo/online.py#L758)）：
```python
else:                                 # 这次 fix 被门控拒了
    n_rej += 1; status = 'gated'
    if chal is not None and |p_meas[:2]-chal[:2]| < chal_m(0.3m) and |Δyaw| < chal_yaw:
        n_chal += 1                   # 与挑战者吻合 → 印证次数+1
    else:
        n_chal = 1; chal = p_meas     # 不吻合 → 丢旧挑战者，用这次 fix 另起一个
    if n_chal >= chal_n(8):           # 连续 8 个被拒 fix 互相吻合
        X = p_meas.copy()             # 夺权：状态换成挑战者（观测）
        P = Rm_reset + diag(0.1², 0.1², 0.003²)
        status = 'reset'; n_rej = 0; chal = None
```
- 第一个被拒 fix → 立为挑战者 `chal`，`n_chal=1`。
- 之后每个被拒 fix 和 `chal` 比：在 `chal_m=0.3 m`、`chal_yaw`（[DEFAULT L588](../odo/online.py#L588)）内 → `n_chal++`；否则重置挑战者、`n_chal=1`。
- `n_chal` 攒够 **`chal_n=8`** → 认定“这 8 个被拒 fix 稳定指向同一新位置，是 X 错了”，**用该 fix 重置状态**，`status='reset'`，滤波跳出错误。

### 防“慢漂移”冒充一致（`chal_fixed=True`）
注释 [L760-L762](../odo/online.py#L760) + [L766-L767](../odo/online.py#L766)：一致性是和**挑战者首帧**（经航位推算对齐）比，不是和“上一帧 fix”比。`chal_fixed=True`（[DEFAULT L591](../odo/online.py#L591)）时吻合后**不**更新挑战者基准——否则一串缓慢漂移的 fix（相邻帧都很近但整体慢慢移）会被误判为一致而错误夺权。固定基准后，只有真正稳定指向同一位置的 fix 才累计。

---

## 4. 三条出口（更新步的分支全景）

更新步按优先级分支（[L748-L772](../odo/online.py#L748)）：

| 条件 | 动作 | status |
|---|---|---|
| 重定位 fix 但 `d2≥gate` 且证据不强 | 拒绝重定位 | `reloc_rejected`（[L748-L749](../odo/online.py#L748)）|
| 重定位 fix 且 `d2≥gate` | 直接 reset 到该 fix | `reloc`（[L750-L752](../odo/online.py#L750)）|
| `d2 < gate` | 标准卡尔曼更新 | `fix`（[L753-L757](../odo/online.py#L753)）|
| `d2 ≥ gate`（普通拒绝） | 累计挑战者；够 8 次则 reset | `gated` / `reset`（[L758-L772](../odo/online.py#L758)）|

> `reloc`（主动）与 `challenger`（被动兜底）是两条逃离错误状态的路：前者丢失后全局检索来的强 fix 直接 reset；后者针对“滤波自信地错着”，靠一致的被拒观测攒够 8 次夺权。
>
> 这里的 `p_meas` 可能来自**常规定位**（用 `X` 查地图 KD-tree 近邻）或**重定位**（MegaLoc 全局检索候选）——两者区别见 [relocalise.md](relocalise.md)。`status='reloc'`/`reloc_rejected` 即重定位路径，其余（`fix`/`gated`/`reset`）是常规路径。

---

## 5. 为什么要这三件事

EKF 的两个危险：**太轻信**视觉（一个错 fix 带飞）、**太固执**（卡错状态出不来）。三招平衡：

1. **门控**（§1）= 默认怀疑，挡掉离谱 fix（防被带飞）。
2. **弱 fix 只更位置**（§2）= 部分采纳，用可信的位置、丢不可信的（少内点）yaw。
3. **挑战者**（§3）= 防固执，当被拒 fix 们持续一致地指向别处，就承认错、跳过去。

**一句话：EKF 更新用马氏门控 χ²=16（按协方差归一化的偏离）判 fix 可信否——可信就卡尔曼更新、否则拒绝；弱 fix（内点<60）把 yaw 测量噪声设爆、只改位置不改朝向（少内点 yaw 不如航向率可靠）；若连续 8 个被拒的视觉观测彼此一致（各 <0.3 m / `chal_yaw`，与挑战者首帧经航位对齐比对以排除慢漂移），判定是状态错了，用这组观测重置状态逃离错误。挑战者是观测，不是 motion 预测。**

---

## 相关文件

| 文件 | 作用 |
|---|---|
| [`odo/online.py`](../odo/online.py) | EKF 更新步（`measure` → `Rm` → 门控/卡尔曼/challenger）、`DEFAULT`（`gate`/`chal_*`/`yaw_min_inl`）|
| [`docs/online_ekf.md`](online_ekf.md) | 测量噪声 `R` = PnP 协方差 ×4 + floor（弱 fix 的 yaw 屏蔽也在这）|
| [`docs/pnp.md`](pnp.md) | `p_meas` 的来源：地图 PnP 观测 |
| [`docs/retrieval.md`](retrieval.md) | 重定位（reloc）用的全局检索候选 |

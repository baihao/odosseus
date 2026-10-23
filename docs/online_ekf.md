# 在线 EKF 的视觉测量噪声 R

> Brief 的那一句：
> *"Its covariance (×4, plus a floor of 0.10 m and 0.23°) is the measurement noise."*
>
> 讲的是在线定位 [`odo/online.py`](../odo/online.py) 的 EKF 更新步里，**视觉 PnP fix** 作为观测时用的测量噪声 `R`
> 是怎么来的。结论：`R` = 「PnP 自己算的协方差」**×4** 再**加一个下限 diag(0.10 m², 0.10 m², 0.23°²)**。

---

## 0. 一句话

测量噪声 `R` 不是单一常数，是三块拼的（[`online.py:722`](../odo/online.py#L722)）：
```python
Rm = cov_meas * cfg['r_scale'] + np.diag([cfg['r0_pos']**2, cfg['r0_pos']**2, cfg['r0_yaw']**2])
#    └ PnP 协方差 ┘   └ ×4 ┘      └──── floor: 0.10 m / 0.10 m / 0.23° ────┘
```
- **covariance** = `cov_meas`：PnP 解自己估计的不确定度，**逐帧动态**算出来。
- **×4** = `r_scale=4.0`：人为放大的保守系数（超参）。
- **floor** = `r0_pos=0.10 m`、`r0_yaw=0.004 rad ≈ 0.23°`：协方差下限（超参）。

参数都在 [`DEFAULT`（L587）](../odo/online.py#L587)：`r_scale=4.0, r0_pos=0.10, r0_yaw=0.004`。

---

## 1. covariance —— PnP 解自己算出来的（动态）

`cov_meas` 来自 [`solve_pnp`](../odo/online.py#L236) 的末尾（[L270-L272](../odo/online.py#L270)）：
```python
s2  = (e[inl]**2).sum() / max(2*n - nd, 1)        # 内点重投影残差的方差 [px²]
cov = inv(Jᵀ J + diag(lam)*s2)[:3, :3] * s2       # (x, y, yaw) 的边际协方差
```
这是标准的**非线性最小二乘协方差估计** `(JᵀJ)⁻¹ × 残差方差`：

- `J` 是重投影残差对位姿的雅可比，`e` 是残差；
- 内点多、几何条件好、残差小 → `cov` 小（这次 fix 可信）；
- 内点少、残差大 → `cov` 大（这次 fix 不可信）。

所以它**逐帧反映这次视觉 fix 的质量**，不是常数。（PnP 失败时直接给巨大协方差 `eye(3)*1e6`，见 [L269](../odo/online.py#L269)。）`measure()` 把 `(pose, cov, info)` 返回（[L577](../odo/online.py#L577)），进入更新步。

---

## 2. ×4 (`r_scale`) —— 把过于乐观的协方差放大

`solve_pnp` 的 `cov` 通常**偏乐观**：它假设残差是独立同分布高斯，忽略了误匹配、相机标定残差、动态场景、地图点误差等系统性因素。直接拿它当 `R`，EKF 会**过度信任视觉**、对离群 fix 太敏感。

所以乘一个保守因子 **×4**（`r_scale=4.0`）把测量噪声放大，让滤波更稳健。这是在 tune lap 上调出来的经验超参。

---

## 3. floor：0.10 m + 0.23° —— 协方差下限

```python
+ np.diag([0.10**2, 0.10**2, 0.004**2])     # r0_pos=0.10 m, r0_yaw=0.004 rad ≈ 0.23°
```
即便某次 PnP 内点特别多、`cov_meas×4` 算出来极小，也强制叠加一个底噪：**位置不确定度至少 0.10 m、朝向至少 0.23°**。

理由：现实里相机标定、地图 3D 点、时钟补偿（`DT_NS`，见 [map_build.md 附录 A](map_build.md)）都有残余误差，再漂亮的 fix 也不可能真到毫米/角秒级。floor 保证 EKF 永远留一点余量，不会被一个“看似完美、实则略偏”的 fix 带跑。

---

## 4. 组合 + 后续按质量再降权

```
R = PnP协方差 × 4   +   diag(0.10², 0.10², 0.23°²)
    └ 逐帧动态 ┘       └──── 固定下限 ────┘
```
拼好后还会按 fix 的可信度进一步放大 `R`（[L724-L732](../odo/online.py#L724)）：

- **弱 fix（内点少）降权**（[L724-L725](../odo/online.py#L724)）：`Rm *= max(1, r_inl_ref / n_inl)` —— 内点越少，`R` 越大。
- **远离赛道的 fix 降权**（[L726-L727](../odo/online.py#L726)）：离建图线越远越不信视觉，`R` 乘 `1 + (map_dist/r_dist_ref)²`。
- **很弱的 fix 只更新位置、不更新 yaw**（[L729-L732](../odo/online.py#L729)）：少量内点的 PnP yaw 不如车端航向率可靠（注释引 Exp24：30 内点的 fix 把 yaw 带偏 1.6° → 漂了 11 m），于是把 yaw 的测量方差设成极大（`radians(30)²`）等效屏蔽。

---

## 5. `R` 在 EKF 更新里的作用

拼好的 `R`（代码里 `Rm`）进入标准 EKF 更新 + 马氏门控（[L733-L748](../odo/online.py#L733)）：
```python
S  = P + Rm                             # 新息协方差 = 预测协方差 + 测量噪声
d2 = nu @ solve(S, nu)                  # 马氏距离²（nu = fix − 预测）
# d2 过大（> gate=16）→ 拒绝这次 fix；否则做卡尔曼更新
```
- `R` 越大（fix 越不可信）→ 卡尔曼增益越小 → 这次 fix 对状态的修正越弱，EKF 更依赖车端航位推算。
- `R` 越小（fix 越可信）→ 增益越大 → 更快把状态拉向视觉 fix。
- 门控 `gate=16`（[`DEFAULT` L588](../odo/online.py#L588)）用 `R` 参与的新息协方差 `S` 判断 fix 是否离谱，离谱就丢弃。

---

## 6. 所以「noise 如何获取」

| 成分 | 来源 | 性质 |
|---|---|---|
| covariance `cov_meas` | [`solve_pnp`](../odo/online.py#L272) 的 `(JᵀJ)⁻¹×残差方差` | **逐帧动态**，反映这次 PnP 质量 |
| ×4 `r_scale` | [`DEFAULT`](../odo/online.py#L587) 超参 | 固定，放大过于乐观的协方差 |
| floor 0.10 m / 0.23° | [`DEFAULT`](../odo/online.py#L587) `r0_pos`/`r0_yaw` | 固定下限，防过度自信 |
| 弱/远 fix 再降权 | [L724-L732](../odo/online.py#L724) | 按内点数、离线距离、yaw 可靠性动态放大 |

**一句话：视觉 fix 的测量噪声 `R` = PnP 自己从雅可比和重投影残差实时算出的协方差（动态，反映本帧 fix 质量）× 4，再加 diag(0.10 m², 0.10 m², 0.23°²) 的下限；×4 与 floor 是 tune lap 调出的保守超参，防止 EKF 过度信任视觉，之后还按内点数/离赛道距离进一步降权。**

---

## 相关文件

| 文件 | 作用 |
|---|---|
| [`odo/online.py`](../odo/online.py) | 在线 EKF：`solve_pnp`（协方差）、`measure`（视觉 fix）、更新步（`R` 组合 + 门控）、`DEFAULT`（`r_scale`/`r0_pos`/`r0_yaw`）|
| [`docs/map_build.md`](map_build.md) | 建图 + `DT_NS` 标定（残余误差是设 floor 的原因之一）|
| [`docs/retrieval.md`](retrieval.md) | 全局检索粗定位 `init`（丢失时的重定位候选）|

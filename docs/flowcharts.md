# 流程图：建图 / 在线预测与更新

两张端到端流程图，配套各环节的详细文档（见每节链接）。

> 说明：为兼容各 Markdown 预览器的 mermaid 解析，节点文字尽量用纯文本（避免 `<` `>` `&` 等特殊字符）。

---

## 1. 建图（offline，一次性）

对应代码 [`odo/map_build.py`](../odo/map_build.py) + 加载过滤 [`odo/online.py` `Map.__init__`](../odo/online.py#L45)。详见 [map_build.md](map_build.md)。

```mermaid
flowchart TD
    A["训练 run mcap 夜间圈"] --> B{"for cam in 7 cams"}
    B --> C{"for frame A in cam"}
    C --> D["SuperPoint 提特征; crop_feats 裁车身 CROP_BOTTOM"]
    D --> E["partner: 前/后各找基线 大于等于 1.5m 的帧 B"]
    E --> F["LightGlue 匹配 A-B; 匹配少于 8 对则弃"]
    F --> G["GT 位姿 at stamp+DT_NS(-9.1ms); GT6.T 得 Tc_A, Tc_B"]
    G --> H["triangulate 前/后两对; 有效性筛查 正深度/重投影/交会角"]
    H --> I["合并: 两对都有取大交会角; 不一致丢弃; 只一对直接用"]
    I --> J["存成功点: kp 像素 + desc 256D fp16 + X 3D"]
    J --> K["帧表: stamp, prev, next, offset, count, T base位姿"]
    K --> C
    C -->|该相机所有帧完| L["保存 cam.npz"]
    L --> B

    subgraph ONLINE["在线加载 online.Map.__init__"]
        M["加载 cam.npz"] --> N["过滤: 时间窗 且 GT pos_accuracy 小于 1m; self/prev/next 都要合格"]
        N --> O["对保留帧 xy 建 KD-tree cKDTree"]
    end
    L -.-> M

    DT["DT_NS = -9.1ms; Exp12 在 L5 扫 dt 取 pos RMSE 最小"] -.-> G
```

### 为什么存这张「帧表」

`<cam>.npz` 把**所有帧**的 3D 点/描述子**拼接成一个大数组**（`X`、`desc`、`kp`，[`map_build.py` L117](../odo/map_build.py#L117)），而不是按帧分开存。所以必须有一张**逐帧的索引 + 元数据表**（[`map_build.py` L109-L111](../odo/map_build.py#L109)），告诉在线端“第 i 帧对应大数组里哪一段、它拍在哪、能不能用”。每个字段都有明确消费者（[`online.py` Map.__init__ L62-L75](../odo/online.py#L62) / `Map.gather` L94-L101）：

| 字段 | 是什么 | 为什么要存（谁用） |
|---|---|---|
| `offset` + `count` | 该帧的点在拼接数组里的**起始下标**和**点数** | 大数组是所有帧点云首尾相接的；`offset/count` 是**切片指针**——取第 i 帧的 3D 点/描述子就是 `X[offset : offset+count]`。没有它无法从一维大数组里还原“哪些点属于哪一帧”（[`Map.__init__` L66-L68](../odo/online.py#L66)、[`Map.gather` L97-L101](../odo/online.py#L97)）。`count>0` 也用来过滤没三角化出点的空帧（[L63](../odo/online.py#L63)）。 |
| `T` (base 位姿) | 该帧拍摄时车的 GT 位姿（4×4） | 多用途：① `T[:,:2,3]` 取 (x,y) 建 **KD-tree** 并做近邻选帧；② `yaw=atan2(T)` 做朝向过滤；③ 定位时把地图 3D 点**投影**到图像、以及给 PnP 提供 z/roll/pitch 基准都要这帧的位姿（[L69-L75](../odo/online.py#L69)）。 |
| `stamp` | 该帧（A）的时间戳 | ① 加载时按**时间窗**过滤（哪段属于当前 fold 的地图）；② 查该时刻的 **GT pos_accuracy** 做 1m 精度过滤与 risk；③ `Map.select` 里按 `stamp` 分「趟(pass)」，每趟取最近帧（[L63-L66](../odo/online.py#L63)、[L87](../odo/online.py#L87)）。 |
| `stamp_prev` / `stamp_next` | 三角化该帧时用的**前/后配对帧**的时间戳（无则 0） | 这帧的 3D 点是 A 和 prev/next 两帧三角化出来的，所以**配对帧的 GT 精度也必须合格**——加载时要求 self + prev + next 三个 stamp 的 `pos_accuracy` 都 <1m，否则整帧丢弃（[L63-L66](../odo/online.py#L63)）。`0` 表示该方向没配对帧，豁免检查。 |

> 一句话：点云用「拼接大数组 + 逐帧 (offset,count) 切片」存，省内存、加载快；`T/stamp/prev/next` 则是把每帧「定位到空间(T)、定位到时间(stamp)、记住它的三角化依赖(prev/next)」，供在线端做 KD-tree 选帧、投影、时间窗/精度过滤。

---

## 2. 在线预测与更新（每帧，causal）

对应 [`odo/online.py`](../odo/online.py) 主循环。详见 [pnp.md](pnp.md)（PnP）、[online_ekf.md](online_ekf.md)（测量噪声 R）、[ekf_update.md](ekf_update.md)（门控/弱 fix/challenger）、[relocalise.md](relocalise.md)（重定位）。

```mermaid
flowchart TD
    START(["每帧 image set at stamp t"]) --> P1["1. PREDICT 航位推算"]

    subgraph PRED["预测步"]
        P1 --> P2["cs.integrate: 车端 v/a/wz 积分得 dx,dy,dth"]
        P2 --> P3["X = move 状态前推"]
        P3 --> P4["P = F P F' + Q; F 运动雅可比 yaw 放大 xy; Q 随距离/转角增长 dr_calibration 标定"]
    end

    P4 --> M1["2. MEASURE 常规定位 measure prior=X, r1约24"]

    subgraph MEAS["视觉测量 每帧"]
        M1 --> M2["Map.select: 用 X 查 KD-tree 近邻; 每相机最多 n_pass=3 帧"]
        M2 --> M3["project 地图3D点得预测像素 uv_m; 引导匹配 半径内 mutual-best 得 matches"]
        M3 --> M4["solve_pnp: Huber GN + 内点重拟合; 得 p_meas, cov_meas, n_inl"]
    end

    M4 --> LOST{"丢失? since 超过 1s 且 常规救不回"}
    LOST -->|是| R1["3. RELOCALISE 重定位"]
    LOST -->|否| UPD

    subgraph RELOC["重定位 丢失时"]
        R1 --> R2["MegaLoc 全局相似度; 取 4 个相隔 8m 以上地图帧的 GT 位姿作 prior"]
        R2 --> R3["对 X 加 4候选 各跑 measure 大半径60; 取内点最多 best"]
        R3 --> R4["best 内点大于等于80 则 p_meas=best, relocated"]
    end
    R4 --> UPD

    subgraph UPDATE["4-5. UPDATE 修正"]
        UPD["Rm = cov_meas 乘4 加 floor 0.10m,0.23deg; 弱fix 内点少于60 屏蔽yaw"] --> U2["nu = p_meas - X; S = P + Rm; d2 马氏距离"]
        U2 --> GATE{"马氏门控 d2 小于 16 ?"}
        GATE -->|是 接受| U3["Kg = P S逆; X += Kg nu; P 下降; status=fix"]
        GATE -->|否 拒绝| U4{"重定位跳变可信? 幅度裕度内 或 内点大于等于300"}
        U4 -->|是| U5["reset X 到候选 status=reloc"]
        U4 -->|否| U6["gated: 累计挑战者; 8个被拒fix互相一致则 reset challenger"]
    end

    U3 --> OUT["输出位姿 at image stamp t; 从 t+dt 补偿回 t"]
    U5 --> OUT
    U6 --> OUT
    OUT --> START
```

---

## 相关文档

| 文档 | 内容 |
|---|---|
| [map_build.md](map_build.md) | 建图 + `DT_NS` 标定 + 加载过滤 (GT 精度 / KD-tree) |
| [retrieval.md](retrieval.md) | 全局检索粗定位 `init`（离线 `qsplit` 版）|
| [pnp.md](pnp.md) | PnP：投影链 / 残差 / 雅可比 / 协方差 / 引导匹配 vs 内点 |
| [online_ekf.md](online_ekf.md) | 测量噪声 `R` = PnP 协方差 ×4 + floor |
| [ekf_update.md](ekf_update.md) | 更新步：马氏门控 / 弱 fix / 挑战者 |
| [relocalise.md](relocalise.md) | 重定位：MegaLoc 候选 prior / 大半径 PnP / 跳变接受 |

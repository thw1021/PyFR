# PyFR 综述报告：模拟能力、数值算法、物理模型与前后处理

> 编写日期：2026-11-07
> 报告对象：本地源码 `/home/tang/packages/PyFR`（develop 分支，commit `2e409712`）＋官方文档（v3.1）
> 引用约定：**[D\*]** 官方文档（含 URL）｜**[P\*]** 论文（本地 Zotero 库）｜**[F\*]** PyFR 论坛帖（含链接/日期/作者）｜**[C\*]** 源码（含文件:行号）｜**[G\*]** Git/GitHub 事实

---

## 0. 资料来源、优先级与核实方法

**资料优先级**（按用户要求）：① 官方文档与源代码 → ② 论文 → ③ 论坛经验（须注明链接/日期/作者）。所有关键结论均标注来源；**事实**（可直接核实）、**验证结论**（经多源交叉验证）、**推断**（明确标注）与**待核实**（无法确认）分别说明。

**资料可访问性核查结果**：
- 本地源码：可访问，git 仓库完整（2 476 个 commit，历史回溯至 2012-09-04 初始提交 [G1]）。
- 本地论文库 `/home/tang/ZoteroData/storage`：可访问，共 6 013 个条目目录，含 `zotero.sqlite`。注：Zotero 桌面程序未运行，Zotero MCP 插件不可用；本报告改为**只读**方式直接查询 `zotero.sqlite`（Python `sqlite3` URI `?immutable=1`）并直接阅读 PDF，未修改库内任何数据。
- 官方文档 <https://pyfr.readthedocs.io/en/latest/>：可访问，39 个用户指南子页全部拉取核实（部分经 `_sources/*.rst.txt` 原始源获取，内容等价）。
- PyFR 论坛 <https://pyfr.discourse.group/>：可访问（Discourse JSON API）。
- GitHub（`api.github.com/repos/pyfr/pyfr`）：可访问。

---

## 1. 概述与版本

### 1.1 PyFR 是什么

PyFR 是开源高阶流场求解器：官方文档原文 *"PyFR 3.1 is an open-source flow solver that uses the high-order flux reconstruction method"* [D1]。它基于**通量重构（Flux Reconstruction, FR）**方法，面向非结构混合网格上的可压缩 Euler / Navier–Stokes 方程，设计目标是在 GPU 等流式架构（streaming architectures）上高效执行高精度 scale-resolving 模拟（LES/uDNS/DNS）[P1 摘要, P2 摘要]。软件采用 Python 外层＋运行时生成低阶代码（C/OpenMP、CUDA、OpenCL、HIP、Metal）的架构，通过 Mako 模板引擎（PyFR-Mako）实现跨平台内核 [P1 §3, D10]。

- **许可证**：3-clause BSD（事实：2014 论文 CPC 程序摘要 *"Freely available under a 3-clause New Style BSD license"* [P1]；2025 论文 *"Licensing provisions: BSD 3-clause"* [P2]；官方文档页面未写明许可证，[D1–D10] 均未提及——待核实文档是否有更新）。
- **开发语言**：Python（外层）＋生成 C（各后端）[P1/P2 程序摘要]。
- **首版**：v0.1.0 于 2013 年末发布 [P2 §1]。

### 1.2 版本与 commit（全部为已核实事实）

| 项目 | 值 | 来源 |
|---|---|---|
| 本地检出 | `develop` 分支，commit `2e409712bbe50ab0e3218a90dfafc78e6013e805`，日期 2026-11-07 | [G1] `git log -1` |
| 与上游关系 | **本地 commit 与上游 `pyfr/pyfr` develop 分支最新提交完全一致**（GitHub API 返回同一 SHA） | [G2] GitHub API |
| 远程仓库 | `origin = git@github.com:thw1021/PyFR.git`（用户 fork，与上游同步） | [G1] `git remote -v` |
| `pyfr/_version.py` | `__version__ = '3.1'` | [C1] `pyfr/_version.py:1` |
| 最近正式发布 | v3.0：2026-02-05；v3.1：2026-03-13（GitHub Releases） | [G3] GitHub Releases API |
| 文档版本 | readthedocs `latest` = **PyFR 3.1**（v3.1 版本文档） | [D1] |
| 本地快照性质 | develop @ `2e409712` = **v3.1 发布之后约 8 个月的开发版**（v3.1 版本号提升于 commit `ddf8c25e`，2026-02-24） | [G1] |

**⚠️ 本地 develop 快照与 v3.1 发布版（即官方文档所描述版本）的重要差异**（事实，git 可查证）：

| 变更 | commit / 日期 | 影响 |
|---|---|---|
| **移除人工压缩性（AC）求解器** `ac-euler` / `ac-navier-stokes`（不可压缩） | `31fcf947` 2026-03-26 "Remove artificial compressibility (AC) solvers." | v3.1 支持 4 个系统，develop 仅剩 `euler`、`navier-stokes` [C2] |
| **移除指数模态滤波** `[soln-filter]` | `15c0713d` 2026-03-26 "Remove exponential modal filter." | v3.1 的 4 种稳定化策略在 develop 中剩 3 种 |
| **新增隐式时间积分框架**（JFNK） | `9593a3a8` 2026-03-26 "Add implicit time stepping infrastructure."（至 HEAD 已有 31 个相关 commit） | develop 独有，未随任何发布版发布 [C10] |
| 后端基础设施重构、显存跟踪、HIP rocBLAS 64-bit GEMM | `160c2cb9`/`1ee328de`/`ecd63452` 2026-03-26 | — |
| **重构 AD（自动微分）基础设施以简化内核编写** | `4e5f0030` "Rework the AD infrastructure to simplify kernel specification." | 对应 `pyfr/dsl` 模块 [C11] |
| 双时间推进（dual-time）随 AC 一并移除 | `31fcf947`（`pyfr/solvers/*dual*`、`dualtime` 字符串在 develop 源码中已无匹配） | v3.1 文档仍描述 `[solver-time-integrator] formulation = dual` [D4]；develop 改为 `formulation = explicit \| implicit` [C10] |

> **报告口径**：凡涉及源码行为，以本地 develop 快照为准并注明；凡涉及官方文档，注明为 v3.1 文档。两者不一致处均在上表与正文标注。

---

## 2. 模拟能力

### 2.1 能力总览（事实）

| 能力维度 | 内容 | 来源 |
|---|---|---|
| 维度 | 2D 与 3D | [P1 Table 1] |
| 单元类型 | tri、quad、tet、pri、pyr、hex（2D：tri/quad；3D：tet/pri/pyr/hex） | [C3] `pyfr/shapes.py:365-612`；[P2 §3.3] |
| 多项式阶 | 任意（Arbitrary） | [P1 Table 1]；`[solver] order` [D3] |
| 网格输入 | Gmsh ASCII `.msh` 2.2 / 4.1（文档所列，推荐 4.1；源码另接受 4.2）；弯曲单元支持至**四阶**、须为**完备拉格朗日**单元 | [D6] meshing；[C4] `pyfr/readers/gmsh.py:252-259` |
| 并行 | MPI（要求 MPI-4）；分区：METIS / SCOTCH / KaHIP；支持加权与按类型平衡分区 | [D2 installation]；[D9 performance_tuning]；[P2 §3.3] |
| 硬件后端 | OpenMP（x86-64/ARM CPU，AVX-512 调优）、CUDA（NVIDIA，计算能力 3.5+）、HIP（AMD/ROCm）、OpenCL（含 AMD/Intel GPU）、Metal（Apple 芯片） | [D2 installation, D10 developer_guide]；[P2 §3.1 Fig.2] |
| 浮点精度 | `single` / `double`（develop 另在隐式预条件中支持 half/bf16 [C10]） | [D3 backend] |
| 跨秩通信 | mpi4py；CUDA-aware / HIP-aware MPI（GPU 指针直接传给 MPI，可用 GPUDirect RDMA） | [P2 §3.1 Multi-node]；[D9] |
| I/O | HDF5 网格/解文件（`.pyfrm`/`.pyfrs`）；插件异步并行写盘；高阶 VTU/pVTU 导出 | [D8 file_format, io]；[P2 §3.7] |
| 在线可视化 | Ascent 插件（Alpine Ascent/Conduit/VTK-m） | [D7] soln-plugin-ascent；[P2 §3.8] |
| 求解器系统 | `euler`、`navier-stokes`（develop）；v3.1 另有 `ac-euler`、`ac-navier-stokes`（人工压缩性不可压缩） | [D3 solver]（v3.1）；[C2] `pyfr/solvers/__init__.py`（develop） |
| 时间积分 | 显式（Euler/TVD-RK3/RK4/RK34/RK45）；v3.1 另有双时间隐式（backward-euler/SDIRK33/SDIRK43）；develop 新增 JFNK 隐式 | [D4]；[C10] |

### 2.2 方程系统（事实）

- **develop（本地）**：`[solver] system` 可取 `euler` | `navier-stokes`（`pyfr/solvers/__init__.py` 仅注册这两个 [C2]）。
- **v3.1**：另支持 `ac-euler` | `ac-navier-stokes`——人工压缩性（artificial compressibility）不可压缩求解器，配合双时间推进、多项式多重网格（`[solver-dual-time-integrator-multip]`）与变局部时间步长 [D3, D4, D5 solver-dual-time-integrator-multip]；该能力在 2025 论文 §3.6 有完整描述 [P2]。
- **历史上主线从未有过**独立的 advection/acoustics/RANS 系统（验证结论：`git log -- pyfr/solvers/advection` 等路径无任何历史；`git log -S "spalart"` 无结果 [G1]；2014 论文 Table 1 仅列 Euler 与可压 NS [P1]）。**注意**：PyFR 框架定位表述为"求解 advection–diffusion 型问题"（数学类别）[P1 标题/摘要]，勿与"内置对流-扩散求解器"混淆。

### 2.3 规模与性能（事实，引自论文）

- 2014：104 块 NVIDIA M2090 GPU 上的可扩展性演示 [P1 摘要/§7]。
- 2016 Gordon Bell 入围：Titan 上 18 000 块 Tesla **K20X**，13.7 DP-PFLOP/s（58% 峰值）[P2 §5.2]。
- 2025（v2.0.3 论文）：超声速 Taylor–Green 涡（7 阶、13 891 500 四面体）强扩展——Frontier：AMD MI250X，至 **2048 GPU rank**（即 1 024 台加速器、每台 2 GCD；未启用 HIP-aware MPI），16→2048 rank 加速比 58.66、效率 0.46；Alps：NVIDIA GH200 至 2048 GPU（**CUDA-aware** MPI），加速比 59.89、效率 0.47。绝对性能：单个 GH200 ≈ **3.7× MI250X 单 GCD**（峰值带宽比仅 2.5，差额源于缓存配置等）；相对 2016 年 K40c（PyFR v0.2.2、四阶四面体网格）单卡 0.122 → 6.004 GDoF/s，提升 **49.2×**（论文口径：硬件+软件合计，因 7 阶每 DoF FLOPs 更多而偏保守），摘要/结论表述为"十年近 50×" [P2 §5.2, Table 2/3]。
- 论坛基准（经验数据）：V100 单卡 p=3 TGV 约 1.61 GDoF/s（含 ×5 变量因子）[F3，fdw 2021-06-04]；GDoF/s 口径含 5 个守恒变量 [F1，fdw 2026-07-24]。

---

## 3. 数值算法

### 3.1 通量重构（FR）方法原理

以下为 FR 方法的标准表述，与 PyFR 2014 论文 §2（Eq. 1–19）[P1] 和 2025 论文 §2（Eq. 1–7）[P2] 一致：

**(1) 问题设定。** 在每个参考单元 $\boldsymbol{\xi}\in\boldsymbol{\Omega}$ 上求解一阶双曲守恒律 $\partial\mathbf{u}/\partial t + \nabla\cdot\mathbf{f}_\alpha = 0$。解用节点基 $\ell^{\zeta^{(u)}}(\boldsymbol{\xi})$ 在**解点（solution points）** $\boldsymbol{\xi}^{(u)}_\zeta$ 展开：

$$u^{\delta}_{\zeta m} = \sum_{\zeta'=0}^{N^{(u)}-1} u^{\delta}_{\zeta' m}\,\ell^{\zeta^{(u)}}_{\zeta'}(\boldsymbol{\xi}^{(u)}_\zeta)\quad(\text{P2 Eq.4})$$

**(2) 通量点与公共通量。** 在单元内部及**界面（面）**上布置**通量点（flux points）**；相邻单元的通量点对在物理空间重合，但各自持有间断的解值 [P1 Fig.2, P1 §2]。在每个界面通量点处用两侧解值构造**公共解（common solution）**与**公共法向通量（common normal flux）**：

$$f^{\delta\,*\,L\zeta^{(f)}}_{nma} = \mathfrak{F}_{ma}\big(u^{\delta\,L\zeta^{(f)}}_{nja},\,u^{\delta\,R\zeta^{(f)}}_{nja},\,\mathbf{n}^{(f)}_{nja}\big)\quad(\text{P2 Eq.5})$$

其中 $\mathfrak{F}_{ma}$ 通常取**黎曼求解器**。法向通量取反号关系 $\mathfrak{F}^{L}=-\mathfrak{F}^{R}$ 保证整体守恒 [P1 Eq.15 附近]。

**(3) 修正（提升）与散度。** 变换域内的连续通量散度由两部分构成：体积项（解点处通量散度的插值/微分）＋把界面通量点处"通量修正量"（$\mathfrak{F}-\text{内部通量}$）经**修正函数** $\mathbf{g}^{\delta\zeta^{(f)}}$ 传播回单元内部的提升项：

$$\big(\nabla\cdot\tilde{\mathbf{f}}\big)^{\delta}_{\zeta m} = \big(\nabla\cdot\tilde{\mathbf{f}}\big)^{\delta\,\mathrm{int}}_{\zeta m} + \sum_{\zeta'=0}^{N^{(f)}-1}\tilde{\mathbf{r}}^{\delta\,(u)}_{\zeta m}\big(\nabla_{\!\boldsymbol{\xi}}\,g^{\delta\,(u)}_{\zeta'}\big)(\boldsymbol{\xi}^{(u)}_\zeta)\,\big(\mathfrak{F}^{\delta\,L\zeta'}_{nma} - \tilde{f}^{\delta\,L\zeta'}_{nma}\big)\quad(\text{P2 Eq.7})$$

修正函数满足 $\mathbf{n}^{(f)}_{nja}\cdot\mathbf{g}^{\delta\zeta^{(f)}}_{ma}(\boldsymbol{\mathbf{X}}^{(f)}_\xi)=\delta_{\zeta\zeta'}$（P1 Eq.8），即在每个通量点处"只对该点响应"。FR 在此框架下统一了 NDG、DG、谱差分等格式 [P1 §1, 引 Huynh 2007]。

**(4) 半离散右端项。** 最终 $\partial\mathbf{u}/\partial t = -J^{-1}_{m n}\big(\nabla\cdot\tilde{\mathbf{f}}\big)^{\delta}_{\zeta m}$（P1 Eq.19），交给时间积分器。

### 3.2 PyFR 中的 FR 实现（develop 源码精读结果）

PyFR 把上述过程编译为少量**稠密矩阵乘**＋**逐点内核**。所有 FR 算子矩阵在 `BaseShape` 中定义 [C5，`pyfr/shapes.py:104-160`]：

| 矩阵 | 定义（源码） | 作用 | 源码位置 |
|---|---|---|---|
| `m0` | `ubasis.nodal_basis_at(fpts)` | 解点→通量点插值（$u_{fpts}$） | `shapes.py:113-115` |
| `m1` | `jac_nodal_basis_at(upts)` 重排 | 解点处 $\nabla_\xi$ 算子（体积散度项 D） | `shapes.py:117-120` |
| `m2` | `norm_fpts ⊗ m0` | 面法向加权×通量点插值（界面"外推"耦合） | `shapes.py:122-125` |
| `m3` | `gbasis_at(upts)` | **修正/提升基**（通量点→解点），即 Eq.7 中 $g$ 的实现 | `shapes.py:127-137` |
| `m4` | `m1` 的梯度重排 | 解点处梯度算子（粘性项未修正梯度） | `shapes.py:139-142` |
| `m6` | `norm_fpts.T ⊗ m3` | 用公共解跳变修正梯度（粘性项） | `shapes.py:144-147` |
| `m7` | `ubasis.nodal_basis_at(qpts)` | 解点→超积分（抗混叠）求积点插值 | `shapes.py:149-151` |
| `m8`/`m9` | `proj_l2(...)` 及其按维度分块 | L2 投影（初值投影；`surf-flux` 抗混叠时投影面通量） | `shapes.py:153-159` |

**修正基 `gbasis` 的构造**（事实）：PyFR（自 2014-11-07 引入 `gbasis_coeffs` 起，commit `3edbb5fe`，早于 v1.0）即把修正函数统一构造成**面δ函数的 L2 投影**，而非按 Huynh/VCJH 逐族实现：对每类面对其配点基在面求积点上积分，得到提升系数（`shapes.py:255-283`；面求积规则 line/quad 用 `gauss-legendre`、tri 用 `williams-shunn` 36 点，`shapes.py:259-264`）。这与 P1 Eq.8–9 的"向量修正函数约束＋Kopriva/Sun 式梯度构造"在离散层面等价（验证结论；P2 Eq.7 的表述即此形式）。

**一个 RHS（`negdivconf` 之前的完整数据流）**——内核注册于 `pyfr/solvers/baseadvec/elements.py:92-110` [C6] 与 `pyfr/solvers/euler/kernels/` [C7]：

```text
scal_upts (解点解)
  │ disu:        M0 · u                    → 解插值到通量点 scal_fpts
  │ qptsu:       M7 · u                    → (仅 anti-alias=flux 且 order>0) 插值到求积点
  │ tflux:       逐点：物理通量 f(u)（euler/kernels/flux.mako）
  │                    经 smats 变换 → 变换通量 f̂（tflux.mako，含线性/弯曲两分支）
  │ intcflux / mpicflux / bccflux: 逐点：内部/跨MPI/边界界面公共法向通量
  │              （内联黎曼求解器宏，euler/kernels/rsolvers/*.mako）
  │ tdivtpcorf:  (M1 − M3·M2)[·M9] · f̂   → 体积散度项（AA 时用求积点通量投影）
  │ tdivtconf:   M3 · f̂*_n               → 界面通量修正（提升）项，累加进同一寄存器
  └ negdivconf:  逐点：−J⁻¹·(∇·F̂) + 源项  → RHS（写入解寄存器）
```

- `tdivtpcorf` 的矩阵表达式 `(M1 - M3*M2)`（抗混叠时再右乘 `M9`）与 P2 Eq.7 "体积项−界面耦合"逐项对应（验证结论）[C6 `elements.py:92-106`]。
- 无粘（euler）：`intcflux/mpicflux/bccflux` 直接取两侧解 `ul/ur` **就地（`inout`）**计算黎曼公共通量；另有一组 `intcent/mpicent/bccent` 内核（`baseadvec/kernels/`＋`euler/kernels/bccent.mako`）对界面两侧做**最小熵阈值 min 归约**（`entmin`），供熵滤波取"单元与 Voronoi 邻域最小熵"之用 [C6, C7; P2 §3.2]。
- 粘性（navier-stokes）：`intconu/mpiconu/bcconu` 计算界面**公共解**（含 `ldg-beta` 加权与周期旋转）写入 `comm_fpts`，供 `M6` 梯度修正；NS 的 `intcflux` 同时取两侧解与梯度（并含人工粘性场 `artvisc`）计算含粘性的公共通量 [C7]。
- 跨 MPI 交换：`BaseAdvectionSystem.__init__` 注册 `scal_fpts` 的 MPI 交换视图（`system.py:28-35` [C8]）；v2.0.3 起 MPI 后端接口可直传 GPU 设备指针（CUDA-Aware / HIP-Aware MPI，GPUDirect RDMA）[P2 §3.1]。

**粘性项（Navier–Stokes）**：`BaseAdvectionDiffusionElements` [C9，`baseadvecdiff/elements.py:21-63`]：
- `tgradpcoru_upts`: `M4 · u` —— 解点处**未修正梯度**；
- `tgradcoru_upts`: `M6 · comm_fpts` —— 用界面**公共解跳变**修正梯度（β 累加），再经 `gradcoru` 逐点内核用 smats/rcpdjac 转为物理梯度（curved/linear 两分支）；
- 通量点处梯度用 `M0` 分块对角结构插值（`gradcoru_fpts`），粘性通量在 `flux.mako` 中合成（含 Sutherland 分支 [C7 `navstokes/kernels/flux.mako:22,75`]）。
- 该"未修正梯度＋公共解跳变提升"正是 P1 §2 Eq.(9)(10)(18) 对梯度与粘性通量的处理方式（验证结论）。

### 3.3 空间离散配置（点集、抗混叠、界面）

**配点/求积规则名**（事实；v3.1 文档 [D5 nodal-point-sets 各页] 与源码一致 [C3-C5]）：

| 形状/界面 | 解点 `soln-pts` | 通量点 `flux-pts` | 求积 `quad-pts` |
|---|---|---|---|
| quad/hex/line | `gauss-legendre`、`gauss-legendre-lobatto` | `gauss-legendre`、`gauss-legendre-lobatto` | 上两者＋`witherden-vincent` |
| tri | `alpha-opt`、`williams-shunn` | `alpha-opt`、`williams-shunn` | `williams-shunn`、`witherden-vincent` |
| tet | `alpha-opt`、`shunn-ham` | — | `shunn-ham`、`witherden`、`witherden-vincent` |
| pri | `alpha-opt~gauss-legendre-lobatto`、`williams-shunn~gauss-legendre(-lobatto)` | — | `williams-shunn~gauss-legendre(-lobatto)`、`witherden-vincent` |
| pyr | `gauss-legendre(-lobatto)` | — | `witherden`、`witherden-vincent` |

求积规则库还含 `alpha-opt`（四面体 α 最优点集）、`witherden-vincent` 高阶规则等，文件位于 `pyfr/quadrules/<shape>/` [C5]。tet 的 `alpha-opt` 即 P2 Fig.1(b)(c) 所述 α-optimised 通量/解点 [P2 §2]。

**抗混叠（anti-aliasing / over-integration）**：`[solver] anti-alias = flux[, surf-flux]`（默认 `none`）[D3, C5 `shapes.py:48-52`]。`flux` 对体积通量使用更高阶求积（通量以求积点求值再 L2 投影，`M9` 路径），`surf-flux` 对面通量超积分。理论依据：配点型投影的混叠误差是不稳定根源，超积分可消除 [P2 §3.2 引 21]；de-aliasing 的系统研究见 Spiegel, Huynh & DeBonis (2015) [P6]。实践指导：p.vincent 在论坛推荐其 AIAA Journal 论文（doi:10.2514/1.J055304）作为超积分阶数选择依据 [F5，2016-08-30/2017-06-26 帖]。代价：三维高阶抗混叠约使运行时间加倍（2025 论文："typically doubled"）[P2 §3.2]。

**黎曼求解器**：`[solver-interfaces] riemann-solver = rusanov | hll | hllc | roe | roem | exact`（v3.1 文档精确取值 [D5 solver-interfaces]）；源码另含 `roe-lm`（LMRoe，注释引用 JCP 230(13), 5263-5287）[C7 `rsolvers/roe-lm.mako:5`]——待核实其对外取值名（文档未列）。正性保护：界面压强下限 `p-min`（默认 $5\epsilon_{mach}$，熵滤波时沿用 `[solver-entropy-filter] p-min`）[C7 `euler/inters.py:14-23`]。

### 3.4 稳定化与激波捕捉

**v3.1 提供四种稳定化策略** [P2 §3.2；D5 对应小节]；**develop 保留后三种**（模态滤波已移除 [G1]）：

| 策略 | 配置节 | 选项 | 原理/出处 |
|---|---|---|---|
| 指数模态滤波（仅 v3.1） | `[soln-filter]` | `nsteps, alpha, order, cutoff` | 对高阶模态指数衰减；保守且便宜，但不区分 Needed 与否 [P2 §3.2; D5 soln-filter] |
| 抗混叠 | `[solver] anti-alias` | `flux, surf-flux` | 求积点求值消混叠 [P2 §3.2; D3] |
| 人工粘性 | `[solver-artificial-viscosity]` | `max-artvisc, s0, kappa` | 基于传感器的高斯核人工粘性，主要防 Gibbs 振荡 [P2 §3.2 引 24; D5 solver-artificial-viscosity] |
| 熵滤波 | `[solver-entropy-filter]` | `d-min, p-min, e-tol, niters, formulation=nonlinear\|linearised`（文档）；另 `f-tol`（仅源码） | 逐单元求满足体积正性与最小熵阈值的滤波强度（原非线性规划，develop 内置逐点求解内核）[P2 §3.2 引 25-28; D5; C12 `baseadvec/entfilter.py`] |

- 熵滤波即 **Dzanic & Witherden (2022) JCP "positivity-preserving entropy-based adaptive filter"** [P5]；2014–2024 年它被 2025 论文的超声速 TGV 对标评为**激波耗散最小**的格式（对比 CODA/FLEXI/SD3D/OpenSBLI，Table 1/Fig.12）[P2 §5.1]。
- 兼容性（事实）：develop 中熵滤波与**隐式时间积分互斥**（`implicit/base.py:36-38` 直接抛错 [C10]）；与**自适应步长**的联用存在收敛性解释困难（论坛技术答复：tdzanic 2024-06-26，[#1115](https://pyfr.discourse.group/t/entropy-filtering-and-adaptive-time-stepping/1115) [F4]）；激波算例（双马赫反射）不建议配双时间，官方建议改显式＋熵滤波（fdw 2025-04-07，[#1244](https://pyfr.discourse.group/t/dual-time-stepping-for-high-mach-flows/1244) [F6]）。

### 3.5 时间积分

**v3.1（官方文档）** [D4]：
- `[solver-time-integrator] formulation = std | dual`。
- `std`：`scheme = euler | rk34 | rk4 | rk45 | tvd-rk3`；`controller = none | pi`（PI 仅 rk34/rk45；`atol/rtol/errest-norm/safety-fact/min-fact/max-fact/dt-max`）。
- `dual`（配合 AC 系统）：`scheme = backward-euler | sdirk33 | sdirk43`；`pseudo-scheme = euler | rk34 | rk4 | rk45 | tvd-rk3 | vermeire`（`vermeire` 为动能稳定伪时间格式，文档未展开）；多重网格级 `[solver-dual-time-integrator-multip]`（`cycle = [(order, nsteps), ...]`）。

**develop（本地源码）** [C10]：
- `formulation = explicit | implicit`（`integrators/__init__.py:16-37`）。
- 显式：stepper `euler / tvd-rk3 / rk4 / rk34 / rk45`（`explicit/steppers.py:12,27,61,194,219`）；`rk34/rk45` 为嵌入式误差估计对（`bhat`，`steppers.py:160-162`），支持 PI 控制器自适应步长（2025 论文 §3.5 "embedded pairs" [P2]）；控制器取值 `none | pi | cfl`（`explicit/controllers.py:62,68,72` 与 `integrators/controllers.py:8,40`）。
- **隐式（v3.1 之后新增，未发布）**：JFNK 框架——stepper `euler(后向欧拉) / trapezium / trbdf2 / kvaerno43`（`implicit/steppers.py:186,194,204,220`）；非线性求解器 `newton / anderson`（`implicit/nonlinear/`）；线性求解器 `gmres / bicgstab / tfqmr`（`implicit/krylov/`）；预条件 `none / block-jacobi`（`precond.py:162,170`），预条件精度可为 `double/single/half/bf16`（`implicit/base.py:25-66`）；雅可比用有限差分扰动（`fd-eps`，`base.py:42-48`）。

### 3.6 内核生成：PyFR-Mako 与 DSL

- **PyFR-Mako**（v3.1 与更早）：逐点内核用 Mako 派生模板语言书写，`<%pyfr:kernel>` 定义内核、`<%pyfr:macro>` 定义宏，渲染为低层代码后编译链接加载；低层语言为受限 C 子集（不允许 if/for/while 语句，仅三元与 `min/max` 等）[D10 developer_guide; P1 §3.5]。
- **DSL（develop）**：`pyfr/dsl/` 模块与 commit `4e5f0030`"Rework the AD infrastructure to simplify kernel specification"表明 3.x 开发线引入了领域专用语言/自动微分机制以简化内核（如 `negdivconf` 等宏的自动生成）[C11; G1]。2025 论文 §3.1 已描述 DSL 支持 `min` 跨数组归约、内核参数 `reduces` 自动归约、GPU 上自动搬运参数至共享内存等 [P2 §3.1 DSL 段]。
- 矩阵乘内核经 **libxsmm**（x86）、**cuBLAS/cuBLASLt**（自动基准选择）、**rocBLAS**、GiMMiK（稀疏小矩阵生成）等执行；数据布局 AoSoA(k) 自动选择 [P2 §3.1 Matrix multiplications / Data layout]。

---

## 4. 物理模型

| 模型 | 状态 | 说明 | 来源 |
|---|---|---|---|
| 可压缩 Euler 方程 | 全版本 | 5 变量理想气体（γ 律），$E = p/(\gamma-1)+\frac12\rho|\mathbf{v}|^2$（P1 Eq.20-21） | [P1 §4.2; C7 euler/kernels/flux.mako] |
| 可压缩 Navier–Stokes | 全版本 | 牛顿流体应力（P1 Eq.22-23），理想气体温压关系（P1 Eq.24） | [P1 §4.3] |
| 粘性/输运 | 全版本 | 常物性（`mu`, `Pr`）或 **Sutherland 定律**（`[solver] viscosity-correction = sutherland`，常数 `cpTref/cpTs`） | [C13 `navstokes/elements.py:19,43`]；[D5 constants] |
| 不可压缩 NS（人工压缩性 AC） | **仅 ≤v3.1**（develop 已移除） | `system = ac-euler / ac-navier-stokes`，常数 `ac-zeta`，BC：`ac-char-riem-inv / ac-in-fv / ac-out-fp / no-slp-wall / slp-wall`；配合双时间＋多项式多重网格 | [P2 §3.6 引 34/36/37]；[D3/D5]（v3.1）；[G1 `31fcf947`] |
| 湍流处理 | 全版本 | **无内置 RANS/LES 模型**（事实：源码与 v3.1 文档均无 Spalart–Allmaras 等条目 [C2/D3/D7]）；定位为 ILES/欠分辨 DNS（uDNS）求解器 [P2 §1/§3.2]；PANS 为研究性扩展（Dzanic et al. 2022 JCP），**非主线功能**（推断，依据 [P5] 论文与主仓库无对应代码） | [P2; P5] |
| 合成湍流入口 | 全版本 | `solver-plugin-turbulence`：合成涡方法（synthetic eddy method），向区域注入各向同性涡，PCG 随机数流，预计算+缓存按截面前缀优化 | [D7 solver-plugin-turbulence; P2 §3.8 引 40-41; P8 Giangaspero 2022 AIAA J] |
| 气动声学（远场） | 全版本 | `soln-plugin-fwh`：Ffowcs Williams–Hawkings 远场噪声近似（匀速移动介质） | [D7 soln-plugin-fwh; P2 §3.8] |
| 体积源项 | 全版本 | `solver-plugin-source`：空间/时间依赖源项注入 | [D7] |
| 壁面力/矩 | 全版本 | `soln-plugin-fluidforce-*`：积分压强与粘性应力→力/矩（升阻力） | [D7; P2 §3.8] |

> **物性/EoS 说明**：状态方程为量热完全气体（理想气体、常 γ）；真实气体效应、多组分等**不**在主线中（未见于 v3.1 文档与源码；带化学反应的 FR 扩展存在于第三方工作，如 Dzanic et al. 2024 可压缩多组分 [P2 引 28]）——待核实是否有合并计划。

---

## 5. 前处理与后处理

### 5.1 工作流与命令（事实 [D9 user_guide]）

```bash
pyfr import mesh.msh mesh.pyfrm          # Gmsh → PyFR 网格
pyfr partition add mesh.pyfrm 4 four     # 添加分区（-e quad:3 -e tri:2 加权 / -e balanced 平衡；分区器 -p metis|scotch|kahip|baseline，文档亦用连写形式 -pmetis）
pyfr region add mesh.pyfrm teapot.stl teapot   # STL 区域（用于 BC/采样/导出）
mpiexec -n 4 pyfr -p run -b cuda mesh.pyfrm case.ini   # 求解（-b 选后端）
pyfr restart mesh.pyfrm soln.pyfrs [new.ini]           # 重启（可换配置）
pyfr export volume --eopt=order:4 mesh.pyfrm soln.pyfrs out.vtu  # 导出（volume/boundary/stl 子命令）
```

- `run/restart/export` 均可 `mpiexec` 并行，但要求网格内存在匹配数量的分区 [D9]；`run/restart` 通用旗标 `-b/--backend`（**必填**）与 `-P/--pname`（选择具名分区）[C14]。`partition` 另有 `list/info/remove/reconstruct` 子命令；develop 还提供 `pyfr upgrade`（升级旧网格/解文件，v3.1 文档未述）[C14]。
- 三类文件：`.ini`（配置）、`.pyfrm`（网格）、`.pyfrs`（解），均为 HDF5 [D9, D8]。

### 5.2 前处理要点（事实 [D6 meshing]）

- Gmsh 体积域必须给物理名 `fluid`；边界面节点须与体网格面**完全一致**，否则 import 抛 `KeyError`（常见原因是体网格尺寸大于面网格点距）。
- 周期边界：物理名 `periodic-*name*-l / -r`，**仅支持平移周期**（单一直角坐标方向）。
- 弯曲单元 ≤ 四阶，须为**完备拉格朗日**（非 serendipity）单元。
- Gmsh 文件内嵌的分区信息会被 PyFR 忽略；分区一律由 `pyfr partition` 重做 [D6, D9]。
- 网格质量：文档未给出显式质量指标（负雅可比在运行期检查：`rcpdjac_at_np` 抛 `Negative mesh Jacobians detected` [C5 `elements.py:354-355`]）。

### 5.3 后处理（事实 [D9 user_guide; D10 post_processing]）

- `pyfr export`：默认导出**高阶 VTK 单元**（阶数=解的阶数）；`--eopt=order:n` 指定高阶单元阶、`--eopt=divisor:n` 细分为线性单元；`-f` 选择字段；`export boundary`（仅 3D）与 `export stl`；批量模式用 `-` 占位符读 stdin 或 `--batchfile`。（v3.1 **无** `gradient/fibres/subset/tessellate` 等导出旗标——梯度由 writer 插件 `write-gradients = true` 写入解文件 [D9/D10]；develop 的 `pyfr export` 还含 `mesh`/`spanwise` 子命令，v3.1 文档仅列 volume/boundary/stl [C14]。）
- ParaView 建议流程 [D10 post_processing]：打开后**始终**跑 *Clean to Grid*（消除 FR 间断表示产生的重复顶点，Point Data Weighting = Average by Number）；高阶显示调 *Nonlinear Subdivision Level*（2–3）；或用 *Tessellate*（自适应细分，可能极耗时）；混连/并行文件加 *Ghost Cells* 消除接缝；大规模用 `.pvtu`＋pvserver/pvbatch 并行（零件数须为渲染进程数的整数倍）。
- 边界/壁面数据（如计算 $C_f$ 需要的法向）：*Clean to Grid* → *Extract Surface* → *Generate Surface Normals*；导出成本可用 writer 的 `region = wall` + `write-gradients = true` 生成子集 `.pyfrs` 再 `export boundary` [D10]。
- 时间平均文件合并：`pyfr tavg merge`；点采样：`pyfr sampler sample --pts=...`；在线渲染：`pyfr ascent render`（各插件页 [D7]）。
- 统计信息：每个 `.pyfrs` 内嵌 `/stats`（INI），含步时/插件计时/MPI 等待时间，`h5dump -d /stats -b --output=stats.ini soln.pyfrs` 提取，用于负载失衡诊断 [D9 performance_tuning]。
- 并行写盘：插件解写默认异步（约 60 s 超时后阻塞等待）；真正写盘用 Python I/O 而非 MPI-IO；**唯一受支持的并行文件系统是 Lustre**（自动配置条带），在 NFS 上并行 I/O 会损坏数据；VTU 导出用 MPI-IO，同理建议单节点或 `.pvtu` [D10 io]。

### 5.4 配置文件（.ini）关键节一览（事实 [D3–D7]）

| 节 | 关键选项 |
|---|---|
| `[backend]` | `precision = single\|double`；`memory-model = normal\|large`；`collect-wait-times` |
| `[backend-openmp]` | `cc, cflags, alignb, schedule`（AVX-512：`cflags = -mprefer-vector-width=512`） |
| `[backend-cuda]` | `device-id = int\|round-robin\|local-rank\|uuid`；`mpi-type = standard\|cuda-aware`；`cublas-nkerns, gimmik-nkerns/nbench, cflags` |
| `[backend-hip]` | 同上（`hip-aware`、`rocblas-nkerns`） |
| `[backend-metal]` / `[backend-opencl]` | `gimmik-*`；OpenCL 另有 `platform-id, device-type, device-id` |
| `[constants]` | `gamma, mu, Pr, cpTref, cpTs, ac-zeta`（AC 系统）及任意自定义常数 |
| `[solver]` | `system, order, anti-alias, shock-capturing, viscosity-correction` |
| `[solver-interfaces]` | `riemann-solver, ldg-beta, ldg-tau`（另 `p-min`：仅源码 `euler/inters.py`，v3.1 文档该页未列） |
| `[solver-elements-<shape>]` / `[solver-interfaces-<kind>]` | `soln-pts / flux-pts / quad-pts / quad-deg / quad-npts`（可加 `-mg-p<order>` 选多重网格级） |
| `[solver-time-integrator]` | v3.1：`formulation = std\|dual`，scheme/controller/pseudo-* 等；develop：`formulation = explicit\|implicit`（隐式另有 `precond, precond-precision, fd-eps`） |
| `[solver-dual-time-integrator-multip]` | `pseudo-dt-fact, cycle`（仅 v3.1/dual） |
| `[solver-entropy-filter]` / `[solver-artificial-viscosity]` / `[soln-filter]` | 见 §3.4 |
| `[soln-ics]` | `rho,u,v,w,p`（含 x,y,z 表达式）；`quad-deg[-<etype>], quad-pts[-<etype>]`（L2 投影初值） |
| `[soln-bcs-<name>]` | `type` = 15 种（v3.1；develop 移除 5 种 ac-* 后为 10 种），详见 §3.4 表与 [D5 soln-bcs] |
| `[soln-plugin-*]` / `[solver-plugin-*]` | 插件（可多实例后缀），见 §6 |

**边界条件全表（v3.1）**[D5 soln-bcs]：Euler 可用 `char-riem-inv`（含变体 `-controlled`/`-mass-flow`/`-pressure`——注：`-controlled`/`-pressure` 见源码 bcs 目录 [C7]，v3.1 文档明确列出 `-mass-flow`）、`slp-adia-wall`、`sup-in-fa`、`sup-out-fn`；NS 另有 `no-slp-adia-wall`、`no-slp-isot-wall`、`sub-in-frv`、`sub-in-ftpttang`、`sub-out-fp`；AC 系统用 `ac-char-riem-inv`、`ac-in-fv`、`ac-out-fp`、`no-slp-wall`、`slp-wall`。（develop 源码核对：euler bcs 8 个文件含 `char-riem-inv-controlled/-mass-flow/-pressure` [C7]；navstokes bcs 与文档一致 [C7]。）

---

## 6. 代码映射（功能/理论项 ↔ 配置项 ↔ 代码 ↔ 出处）

> 代码列均相对 `/home/tang/packages/PyFR`（develop @ `2e409712`）；"v3.1 差异"栏给出与官方文档的差异。

| 功能/理论项 | 配置项（.ini） | 代码模块/文件 | 出处 |
|---|---|---|---|
| FR 解点/通量点/求积点布置 | `[solver-elements-*] soln-pts`、`[solver-interfaces-*] flux-pts`、`quad-*` | `pyfr/shapes.py`（BaseShape：upts/fpts/qpts/gbasis） | [C5; D5 nodal-point-sets] |
| 解→通量点插值（FR 第一步） | — | `M0`：`shapes.py:113`；内核 `disu`：`solvers/baseadvec/elements.py:93` | [C5,C6; P1 Eq.6] |
| 变换通量 | — | `solvers/euler/kernels/tflux.mako`、`flux.mako`（NS 粘性部分在 `navstokes/kernels/flux.mako`） | [C7; P1 Eq.12-13] |
| 公共法向通量（黎曼求解） | `[solver-interfaces] riemann-solver, p-min` | `euler/kernels/{intcflux,mpicflux,bccflux}.mako`＋`rsolvers/{rusanov,hll,hllc,roe,roem,roe-lm,exact}.mako`；`euler/inters.py:14-23` | [C7; P1 Eq.15-16; D5] |
| 体积散度项（FR 第二步） | — | `tdivtpcorf`：`(M1−M3·M2)(·M9)`，`baseadvec/elements.py:92-106` | [C6; P2 Eq.7] |
| 修正/提升项 | — | `M3`（修正基，L2 投影构造 `shapes.py:255-283`）；`tdivtconf`：`elements.py:107-110` | [C5,C6; P1 Eq.8-9] |
| RHS 组装（含 −J⁻¹、源项） | `[solver-plugin-source]` 等 | `negdivconf.mako`（`baseadvec/kernels/`）、`evalsrcmacros.mako` | [C6; P1 Fig.3] |
| 粘性梯度重构 | `ldg-beta/ldg-tau`（文档节 `[solver-interfaces]`） | `M4/M6`：`shapes.py:139-147`；`baseadvecdiff/elements.py:44-63`；`kernels/gradcoru.mako` | [C9; P1 Eq.9-10,18] |
| 抗混叠（超积分） | `[solver] anti-alias = flux[,surf-flux]` | `shapes.py:48-52,131-136`；`qptsu/tdivtpcorf M9 路径` | [C5,C6; P2 §3.2; P6] |
| 熵滤波激波捕捉 | `[solver] shock-capturing = entropy-filter`＋`[solver-entropy-filter]` | `baseadvec/entfilter.py`；内核 `euler/kernels/entropy{,filter,local}.mako` | [C12; P5; D5] |
| 人工粘性 | `shock-capturing = artificial-viscosity`＋`[solver-artificial-viscosity]` | `baseadvecdiff/artvisc.py`；`kernels/{artvisc,avfill,shocksensor}.mako` | [C9; P2 §3.2; D5] |
| 指数模态滤波 | `[soln-filter]`（**仅 v3.1**） | v3.1 存在；develop 已删（`15c0713d`） | [D5; G1] |
| 时间积分（显式） | `[solver-time-integrator] formulation=explicit(v3.1: std)`，`scheme, controller` | `integrators/explicit/{steppers,controllers}.py` | [C10; D4] |
| 自适应步长（嵌入式 RK 对） | `controller = pi`＋`atol/rtol/...` | `explicit/steppers.py:160-194+`（rk34/rk45 `bhat`） | [C10; P2 §3.5] |
| 双时间（v3.1，AC 系统） | `formulation = dual`＋`[solver-dual-time-integrator-multip]` | v3.1 `integrators/dual/*`；develop 已删 | [D4/D5; G1] |
| 隐式 JFNK（develop 新增） | `formulation = implicit`，`scheme = trapezium\|trbdf2\|kvaerno43...`，`precond` | `integrators/implicit/{base,steppers,controllers,precond}.py`、`krylov/{gmres,bicgstab,tfqmr}.py`、`nonlinear/{newton,anderson}.py` | [C10; G1 `9593a3a8`] |
| CFL 控制 | （显式控制器） | `explicit/controllers.py:68`（`CFLControllerMixin`） | [C10] |
| 初值（含 L2 投影） | `[soln-ics]`＋`quad-deg/quad-pts` | `solvers/base/elements.py:108-151`（`proj_l2` → `M8`） | [C5; D5 soln-ics] |
| 边界条件 | `[soln-bcs-*] type` | `euler/kernels/bcs/*.mako`、`navstokes/kernels/bcs/*.mako` | [C7; D5] |
| Sutherland 粘性 | `[solver] viscosity-correction = sutherland` | `navstokes/elements.py:19,43`；`flux.mako:22,75` | [C13; D5 constants] |
| 界面/MPI 交换 | —（自动按分区） | `solvers/base/inters.py`、`baseadvec/system.py:28-42`（`register_mpi_exchange`）、`mpiutil.py` | [C8; P2 §3.1] |
| 后端（内核生成/编译） | `[backend]`、`[backend-<name>]` | `backends/{openmp,cuda,hip,opencl,metal}/`、`backends/base/`（`kernel_generator`、`pointwise`） | [D10; P2 Fig.2] |
| 矩阵乘加速 | `gimmik-*`、`cublas-nkerns` 等 | `backends/cuda/{gimmik,cublaslt}.py`、`hip/rocblas.py`、OpenMP libxsmm | [C15; P2 §3.1] |
| 网格导入 | `pyfr import`（`-t <type>`、`-l <tol>` 线性化容差、`--no-renumber/--no-reorient`） | `readers/{gmsh,native,stl}.py`、`reorient.py`、`shared_nodes.py` | [C4; D6] |
| 分区 | `pyfr partition add [-e shape:weight\|balanced] [-pmetis\|-pscotch\|-pkahip]` | `partitioners/{metis,scotch,kahip,baseline}.py`、`renumber.py`、`reconstruct.py` | [C14; D9] |
| 解写出（异步/并行/子集） | `[soln-plugin-writer]`（`write-gradients, region, async-timeout` 等） | `plugins/soln/writer.py`；HDF5：`writers/native.py` | [D7,D10 io] |
| VTU 导出 | `pyfr export volume\|boundary\|stl` | `writers/vtk/{base,volume,boundary,stl,spanwise,output}.py` | [C16; D10] |
| 时间平均/采样/力/残差/NaN 检查/积分/FWH/Ascent | `[soln-plugin-tavg/sampler/fluidforce-*/residual/nancheck/integrate/fwh/ascent/dtstats/pseudostats]` | `plugins/soln/*.py` | [D7] |
| 源项/合成湍流 | `[solver-plugin-source/turbulence]` | `plugins/solver/{source,turbulence}.py`（turbulence：boostree RTree＋PCG32） | [C17; D7; P2 §3.8] |
| 触发器（插件节内 `trigger-*`） | 插件节内 `trigger-{time,wallclock,signal,file,expression,duration,manual}` | `plugins/triggers/` | [C18] |
| 命令行入口 | `pyfr <cmd>` | `pyproject.toml [project.scripts]`；`pyfr/__main__.py` | [C14] |
| 几何度量（守恒度量/线性元优化） | —（自动） | `solvers/base/elements.py:413-475`（Kopriva 不变形式 smats，注释引 JSC 26(3), 301-327 Eq.37） | [C5; P2 §3.4] |
| DSL/AD 内核生成（develop） | —（开发态） | `pyfr/dsl/` | [C11; G1 `4e5f0030`; P2 §3.1] |
| 性能（任务图/缓存分块/AoSoA） | —（自动/后端级） | `backends/base/`（task graph、`autotune-ifac`） | [P2 §3.1; C15] |

**插件清单**（develop；名称即 `[soln-plugin-<name>]` / `[solver-plugin-<name>]`，事实 [C18; D7]）：
solution 类：`ascent, dtstats, fluidforce, fwh, integrate, nancheck, pseudostats, residual, sampler, tavg, writer`；solver 类：`source, turbulence`。（v3.1 文档所列与此一致 [D7]。）

---

## 7. 验证与限制

### 7.1 公开验证与对标（验证结论，均给出处）

1. **文档示例集**（可复现工作流）：2D Euler Vortex、2D Double Mach Reflection、2D Couette、2D 不可压圆柱（v3.1）、2D Viscous Shock Tube、3D Triangular Aerofoil（含合成湍流入口）、3D Taylor–Green；测试用例库 `github.com/PyFR/PyFR-Test-Cases` [D11 examples]。文档明确提示示例规模小，**不适合**做扩展性/性能研究 [D11]。
2. **超声速 Taylor–Green 涡对标**（v2.0.3 论文，Ma=1.25、Re=1600，与 Chapelier 等 5 个求解器比较）：PyFR（FR4＋熵滤波）在 64³–512³ DoF 下耗散最低、激波分辨最锐利，与参考 TENO 数据吻合 [P2 §5.1, Table 1, Fig.11-12]。
3. **扩展性**：Frontier（MI250X，未用 HIP-aware MPI）16→2048 rank（1 024 台加速器×2 GCD）加速比 58.66、效率 0.46；Alps（GH200，CUDA-aware MPI）至 2048 GPU 加速比 59.89、效率 0.47 [P2 Table 2/3]；早期 Titan 18 000 块 K20X 13.7 DP-PFLOP/s（58% 峰值，2016 Gordon Bell 入围）[P2 §5.2]。
4. **应用记录**（2025 论文 §4 综述，均给出参考文献编号）：MTU T106/T16 低压涡轮叶栅 LES、Arup 高层建筑风工程（不可压）、火星直升机旋翼、NASA 超声速逆推、风电机组、热声发动机、机翼噪声等 [P2 §4]。

### 7.2 已知限制与注意事项（事实＋来源；推断已标注）

| 限制 | 说明 | 来源/性质 |
|---|---|---|
| 金字塔单元仅支持仿射基底 | 底面须为平行四边形；任意四棱锥需有理多项式空间，PyFR 未实现 | 事实：[F2 WillT 2021-11-22](https://pyfr.discourse.group/t/errors-on-mesh-with-pyramid-elements/501)；[P2 §3.3 "partial support for pyramids... affine"] |
| 仅平移周期边界 | 单一直角坐标方向 | 事实：[D6 meshing] |
| OpenCL（NVIDIA）退出时段错误 | NVIDIA OpenCL 子缓冲长期 bug，数据已写完，结果不受影响 | 事实：[D2 installation] |
| 熵滤波 vs 隐式/自适应步长 | develop 中与隐式互斥（抛错）；与自适应步长联用使误差来源难以归因 | 事实：[C10 `implicit/base.py:36-38`]；经验：[F4 tdzanic 2024-06-26] |
| 双时间＋激波算例不可行（v3.1 语境） | 官方答复：激波算例应用熵滤波＋显式 | 经验：[F6 fdw 2025-04-07] |
| 并行 I/O 仅支持 Lustre | NFS 上并行写会损坏数据；VTU 导出同理 | 事实：[D10 io] |
| SMT 无收益；OpenMP 扩展性依赖异步 MPI progression；AVX-512 需手工开 | 调优建议 | 事实：[D9 performance_tuning] |
| develop 移除 AC/双时间/模态滤波 | 依赖不可压求解或模态滤波的用户停留在 v3.1 及更早版本 | 事实：[G1]；影响评估为**推断** |
| HIP-aware MPI 缺失限制 Frontier 扩展性 | 2048 rank 时两平台 scaling 相当，但 GH200 基线更高使跨互联需多传约 3.7× 数据，故瓶颈更可能是 Frontier 上无法启用 HIP-aware MPI 及 HIP runtime/graphs 的未决问题（作者自述为推测） | 验证结论（论文推测）：[P2 §5.2] |
| 技能门槛 | 作者强调 SRS 需要网格/后处理/分析训练 | 验证结论：[P2 §6] |
| 版本对齐 | 官方文档 latest=v3.1，滞后于 develop（隐式积分、DSL 重构等未见于文档） | 事实：[D1; G1] |
| 文档未写许可证；`pyfr export` 的 `--eopt` 取值细节（如 `order` 上限）随版本变化 | 需以 `pyfr --help` 为准 | 待核实：[D1-D10] |

### 7.3 事实/验证结论/推断标注汇总

- **事实**：全部版本/commit 信息、配置项取值、代码文件与行号、论坛发言原文（链接+日期+作者）、论文数据。
- **验证结论**：FR 数学流程与 PyFR 内核的逐项对应（P1/P2 公式 vs `shapes.py`/kernels 双重核对）；"PyFR 长期主线只有 euler/navier-stokes(+AC)"（git 全历史核对）；熵滤波耗散最低（2025 论文对标，属论文结论）。
- **推断**：PANS 为研究扩展而非主线；develop 移除 AC 对用户工作流的影响；`roe-lm` 是否可通过 `.ini` 直接选择（文档未列，源码存在）。
- **待核实**：官方文档是否会跟进 develop 的新特性；3.x 是否计划重新引入不可压缩求解器。

---

## 8. 参考文献

### 8.1 官方文档（v3.1，readthedocs latest）

- [D1] 总览 <https://pyfr.readthedocs.io/en/latest/>（引文："PyFR 3.1 is an open-source flow solver..."）
- [D2] 安装 <https://pyfr.readthedocs.io/en/latest/installation.html>
- [D3] `[backend]`/`[backend-*]` <https://pyfr.readthedocs.io/en/latest/backends/backend.html>（及 backend-openmp/cuda/hip/metal/opencl 各页）；`[solver]` <https://pyfr.readthedocs.io/en/latest/systems/solver.html>；`[constants]` <https://pyfr.readthedocs.io/en/latest/systems/constants.html>
- [D4] 时间积分 <https://pyfr.readthedocs.io/en/latest/systems/solver-time-integrator.html>；多重网格 <https://pyfr.readthedocs.io/en/latest/systems/solver-dual-time-integrator-multip.html>
- [D5] 界面/黎曼 <https://pyfr.readthedocs.io/en/latest/systems/solver-interfaces.html>；熵滤波 <.../systems/solver-entropy-filter.html>；人工粘性 <.../systems/solver-artificial-viscosity.html>；解滤波 <.../systems/soln-filter.html>；初值 <.../boundary-initial-conditions/soln-ics.html>；边界 <.../boundary-initial-conditions/soln-bcs.html>；配点 <.../nodal-point-sets/solver-elements-{quad,tri,hex,tet,pri,pyr}.html> 与 <.../nodal-point-sets/solver-interfaces-{line,quad,tri}.html>
- [D6] 网格 <https://pyfr.readthedocs.io/en/latest/meshing.html>
- [D7] 插件 <https://pyfr.readthedocs.io/en/latest/plugins/soln-plugin-writer.html> 等 13 个插件页
- [D8] 文件格式 <https://pyfr.readthedocs.io/en/latest/file_format.html>；磁盘 I/O <https://pyfr.readthedocs.io/en/latest/io.html>
- [D9] 用户指南 <https://pyfr.readthedocs.io/en/latest/user_guide.html>；性能调优 <https://pyfr.readthedocs.io/en/latest/performance_tuning.html>
- [D10] 后处理 <https://pyfr.readthedocs.io/en/latest/post_processing.html>；开发者指南 <https://pyfr.readthedocs.io/en/latest/developer_guide.html>
- [D11] 示例 <https://pyfr.readthedocs.io/en/latest/examples.html>

### 8.2 论文（本地 Zotero 库，附 PDF 路径）

- [P1] F.D. Witherden, A.M. Farrington, P.E. Vincent, *PyFR: An open source framework for solving advection–diffusion type problems on streaming architectures using the flux reconstruction approach*, **Computer Physics Communications** 185 (2014) 3028–3040. doi:10.1016/j.cpc.2014.07.011 ／PDF: `/home/tang/ZoteroData/storage/YFS4X9TJ/`
- [P2] F.D. Witherden, P.E. Vincent, W. Trojak, et al., *PyFR v2.0.3: Towards industrial adoption of scale-resolving simulations*, **Computer Physics Communications** 311 (2025) 109567. doi:10.1016/j.cpc.2025.109567 ／PDF: `/home/tang/ZoteroData/storage/CNPICWY5/`
- [P3] F.D. Witherden, *Python at Petascale with PyFR*, **Computing in Science & Engineering** 23 (2021). ／PDF: `/home/tang/ZoteroData/storage/XGNIVFZT/`
- [P4] F.D. Witherden, B.C. Vermeire, P.E. Vincent, *Heterogeneous computing on mixed unstructured grids with PyFR*, **Computers & Fluids** (2015). ／PDF: `/home/tang/ZoteroData/storage/YB6M8CV5/`
- [P5] T. Dzanic, F.D. Witherden, *Positivity-preserving entropy-based adaptive filtering for discontinuous spectral element methods*, **J. Comput. Phys.** (2022) ／`UI2GFAHV`；及 T. Dzanic, S.S. Girimaji, F.D. Witherden, *PANS simulations within a high-order FR framework*, **JCP** (2022) ／`DWIPNVG4`
- [P6] S.C. Spiegel, H.T. Huynh, J.R. DeBonis, *De-aliasing through over-integration applied to the flux reconstruction and discontinuous Galerkin methods* (2015) ／`NWP4YWCD`
- [P7] N.A. Loppi, F.D. Witherden, A. Jameson, P.E. Vincent, *Locally adaptive pseudo-time stepping for high-order flux reconstruction*, **JCP** (2019) ／`T96NCQBM`
- [P8] G. Giangaspero, F. Witherden, P. Vincent, *Synthetic turbulence generation for high-order scale-resolving simulations on unstructured grids*, **AIAA Journal** (2022) ／`HGGMBQD2`
- [P9] W. Trojak, N.R. Vadlamani, J. Tyacke, F.D. Witherden, A. Jameson, *Artificial compressibility approaches in flux reconstruction for incompressible viscous flow simulations*, **Computers & Fluids** (2022) ／`LH2FVYSQ`
- [P10] S. Akkurt, F. Witherden, P. Vincent, *Cache blocking for flux reconstruction: Extension to Navier–Stokes equations and anti-aliasing*, **CPC** (2024) ／`ALC27J5M`
- [P11] R. Vandenhoeck, A. Lani, *Implicit high-order flux reconstruction solver for high-speed compressible flows*, **CPC** (2019) ／`Z7VF4MY6`
- [P12] F.D. Witherden, P.E. Vincent, *On the development and implementation of high-order flux reconstruction schemes for CFD*（技术报告，Imperial College, 2015）／`N5MZ4GGA`
- [P13] F.D. Witherden et al., *Chapter 10: High-Order Flux Reconstruction Schemes*（手册章节, 2016）／`ZCNATNAJ`
- [P14] F. Wang, *On the implicit LES of turbomachinery flows using the flux reconstruction method*, **Computers & Fluids** (2024) ／`6WYVY8M8`

> 关键 FR 原始文献（经 P1/P2 转引）：H.T. Huynh, *A flux reconstruction approach to high-order schemes including discontinuous Galerkin methods*, AIAA 2007-4079；C.G. Vincent, P. Castonguay, A. Jameson, *A new class of high-order energy stable flux reconstruction schemes*, J. Sci. Comput. (2011)。（本地库未见 Huynh 2007 原文——待核实；引文以 P1/P2 文献表为准。）

### 8.3 论坛帖（链接＋日期＋作者）

- [F1] *PyFR GPU Performance* — <https://pyfr.discourse.group/t/pyfr-gpu-performance/1369>（2026-07-21 起；j1woo 提问，**fdw**［Freddie Witherden］2026-07-22/24 答复：对标论文用 order=7；GDoF/s 计入 ×5 变量）
- [F2] *Errors on mesh with pyramid elements* — <https://pyfr.discourse.group/t/errors-on-mesh-with-pyramid-elements/501>（2021-11-22；Zhenyang 提问，**WillT**［Will Trojak］答复：仅支持仿射底面金字塔）
- [F3] *TGV Performance Numbers* — <https://pyfr.discourse.group/t/tgv-performance-numbers/407>（2021-06-03；**WillT** 发帖，**fdw** 2021-06-04：V100 p=3 ≈1.61 GDoF/s）
- [F4] *Entropy filtering and adaptive time stepping* — <https://pyfr.discourse.group/t/entropy-filtering-and-adaptive-time-stepping/1115>（2024-06-26；Frankx9 提问，**tdzanic**［Tarik Dzanic］答复兼容性原因）
- [F5] *Options Anti-Aliasing in PyFR* — <https://pyfr.discourse.group/t/options-anti-aliasing-in-pyfr/116>（2016-08-30；Antonio_Garcia-Uceda 提问；**p.vincent** 2017-06-26 给出 AIAA J055304 论文）
- [F6] *Dual time stepping for high Mach flows* — <https://pyfr.discourse.group/t/dual-time-stepping-for-high-mach-flows/1244>（2025-03-23；Olaf 提问，**fdw** 2025-04-07：激波建议熵滤波＋显式）
- [F7] *Running PyFR with OpenMP* — <https://pyfr.discourse.group/t/running-pyfr-with-openmp/1180>（2024-10-21；含 `PYFR_DEBUG_OMP_DISABLE_CACHE=1` 排障）
- [F8] *Incompressible BCs not fully enforced* — <https://pyfr.discourse.group/t/incompressible-bcs-not-fully-enforced/786>（2023-04-03；AC 不可压设置讨论）
- [F9] *Error: All viewed matrices must belong to the same allocation extent* — <https://pyfr.discourse.group/t/error-all-viewed-matrices-must-belong-to-the-same-allocation-extent/1363>（2026-05-20；julius 报告，**fdw** 2026-05-23 修复入 develop——develop 快照已含该修复）
- [F10] *The turbulent boundary layer does not collapse onto the log-law region* — <https://pyfr.discourse.group/t/the-turbulent-boundary-layer-does-not-collapse-onto-the-log-law-region/1374>（2026-10-28；合成湍流＋ y⁺ 讨论）
- [F11] *PyFR release: v3.1* — <https://pyfr.discourse.group/t/pyfr-release-v3-1/1347>（2026-03-13；**p.vincent** 发布公告：Ascent/point-sampler/FWH bug 修复）

### 8.4 代码与 Git/GitHub（[C\*] 汇总）

- [C1] `pyfr/_version.py`；[C2] `pyfr/solvers/__init__.py`；[C3] `pyfr/shapes.py`（形状类）；[C4] `pyfr/readers/gmsh.py`；[C5] `pyfr/shapes.py`（opmat/M0–M9/gbasis，:104-283）；[C6] `pyfr/solvers/baseadvec/elements.py`（:92-110 内核注册）；[C7] `pyfr/solvers/euler/kernels/`、`pyfr/solvers/navstokes/kernels/`；[C8] `pyfr/solvers/baseadvec/system.py`、`pyfr/solvers/base/inters.py`；[C9] `pyfr/solvers/baseadvecdiff/{elements.py,artvisc.py}`；[C10] `pyfr/integrators/`（explicit/implicit/controllers/steppers）；[C11] `pyfr/dsl/`；[C12] `pyfr/solvers/baseadvec/entfilter.py`；[C13] `pyfr/solvers/navstokes/elements.py`；[C14] `pyfr/__main__.py`、`pyproject.toml`；[C15] `pyfr/backends/`（各后端 `gimmik.py`/`cublaslt.py`/`rocblas.py`/`compiler.py`）；[C16] `pyfr/writers/vtk/`；[C17] `pyfr/plugins/solver/{source,turbulence}.py`；[C18] `pyfr/plugins/`（soln/solver/triggers/postproc/cli 子包）
- [G1] 本地 git 仓库（`/home/tang/packages/PyFR`）：`git log`（commit `2e409712`，2026-11-07；关键提交 `ddf8c25e` 2026-02-24 版本号提升 v3.1、`31fcf947`/`15c0713d`/`9593a3a8`/`160c2cb9`/`4e5f0030` 2026-03-25/26 重构与移除；全历史 2 476 commits）
- [G2] GitHub API `repos/pyfr/pyfr/commits/develop`（develop HEAD = `2e409712...`，2026-11-07T09:08:25Z）
- [G3] GitHub Releases `repos/pyfr/pyfr/releases`（v3.0 2026-02-05；v3.1 2026-03-13；v2.x 全序列）

---

### 附：主要未决事项（待核实清单）

1. `roe-lm` 黎曼求解器在 `.ini` 中的对外取值名（源码存在、v3.1 文档未列）。
2. 官方文档对 develop 新特性（隐式积分、DSL/AD、后端重构）的跟进时间。
3. 官方文档页面对许可证的正式表述（目前仅论文与 GitHub 仓库声明 BSD-3）。
4. 本地 Zotero 库中是否还有其他 PyFR 直接相关文献未被本报告收录（已按 6 组关键词检索）。

---

## 9. 复核记录（2026-11-08）

本报告经第二轮系统复核：源码断言逐条重验（内核模板、CLI、积分器、后端选项），v3.1 文档经**直接抓取的页面 HTML** 复核（backend、backend-cuda、installation、user_guide、solver、solver-interfaces、time-integrator、entropy-filter、artificial-viscosity、soln-bcs、meshing、post_processing、performance_tuning、io、examples、nodal-point-sets tet/pri），论文断言经 `pdftotext` 对 P2 第 10–11 页复核，论坛断言经已存档的线程 JSON 复核。结果：

**复核通过（抽样/全查无误）**：全部版本/commit 断言 [G1–G3]；M0–M9 定义与行号；`disu/qptsu/tdivtpcorf/tdivtconf/negdivconf` 数学语义（`negdivconf` 源码确认 `out = -rcpdjac*tdivtconf + src`）；隐式 stepper 类名（`ImplicitEulerStepper` 确为后向欧拉）与 Krylov/非线性/预条件名；`[solver]`/`[solver-interfaces]`/熵滤波/人工粘性/时间积分/边界条件（15 种）/配点（tet/pri 逐值）/backend 选项（backend、backend-cuda 逐项）/安装依赖（CUDA≥3.5、MPI-4、ROCm≥6.4.1、GCC≥12、libxsmm commit 钉扎、分区器版本）/性能调优（AVX-512 旗标、SMT、NUMA、异步 progression、ε≈0.8）/I/O（Lustre、异步 60 s、NFS 警告）/示例清单与警告——均与文档原文逐字相符；`pyfr export --eopt=order:n/divisor:n`、批量模式、`partition add/remove/reconstruct`（含 `-p`/`-e` 语法）与源码一致；论坛 11 帖的链接/日期/作者与原文一致。

**复核发现并已修正的错误（9 处）**：
1. ~~"Titan 上 18 000 块 Tesla P100"~~ → **Tesla K20X**（P2 §5.2 原文）。
2. ~~"性能提升 49×（其中 16× 来自硬件）"~~ → 论文无"16× 硬件"表述（疑为与"16 KiB per CU"缓存数字混淆）；改为 49.2×＝硬件+软件合计、十年"近 50×"。
3. ~~"单卡 GH200 ≈ 3.7× MI250X"~~ → 单个 GH200 ≈ 3.7× MI250X **单 GCD**（带宽比仅 2.5）。
4. ~~Alps"hip-aware MPI"~~ → Alps 用 **CUDA-aware** MPI（HIP-aware MPI 在 Frontier 上不可用才是瓶颈话题）。
5. ~~Frontier"至 2048 GPU"~~ → 至 2048 GPU rank（=1 024 台 MI250X×2 GCD）。
6. ~~§7.2"~5.7× 互联时间差"~~ → 论文数字为跨互联多传约 3.7× 数据。
7. ~~"界面公共解先由 intconu/mpiconu/bcconu 计算…再用于 intcflux 的黎曼求解"~~ → euler 实为 `intcflux` 直取两侧解算黎曼通量，`intcent/mpicent/bccent` 是熵滤波的 entmin 归约；`intconu/mpiconu/bcconu`（公共解）为 NS 专用。
8. ~~"`pyfr import`（`--ltol`）"~~ → 实际旗标为 `-t`/`-l`/`--no-renumber`/`--no-reorient`；并补充 `run/restart` 的 `-b`（必填）与 `-P`，develop 的 `export mesh/spanwise` 与 `pyfr upgrade`。
9. 文献交叉引用修正：PANS→[P5]、Giangaspero→[P8]；"修正基自 3.x 改为 L2 投影"→ 自 2014-11（commit `3edbb5fe`，早于 v1.0）即是；若干精确化（Gmsh 4.2 为源码扩展、`p-min`/`f-tol` 标注"仅源码"、`qptsu` 需 `order>0`、`§2.4` 空引用、控制器全集 `none|pi|cfl`）。

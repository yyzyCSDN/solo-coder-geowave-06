# geowave.multiple — 海上自由表面多次波压制 (SRME)

针对海上数据自由表面多次波覆盖目标层、简单带通伤及一次反射的场景，
提供完整的表面相关多次波预测 + 自适应相减流程。

## 流程

```
缺炮缺道数据 ──► 规则化(POCS) ──► 多次波预测(反馈模型) ──► 分窗自适应相减 ──► 一次波
                  regularize.py    predict.py               subtract.py        (保留信号)
```

1. **规则化** `regularize.regularize_cube`：SRME 预测要求炮、检波点位于同一
   表面规则网格，缺炮/缺道会在模型中产生空洞。先线性插值给初值，再用
   POCS（Abma & Kabir 2006，(炮,道) 二维谱域逐次阈值收缩）重建缺失道，
   观测样点每轮复原、不被改动。支持随机丢道、整炮缺失、整缆缺失。

2. **预测** `predict.predict_multiples`：反馈模型（Verschuur et al. 1992），
   频率域 M(ω) = −P(ω)@D(ω)（自由表面反射系数 −1），迭代 P ← D + P·D
   收敛到 P = D(I−D)⁻¹，`n_iter ≥ 2` 时高阶多次波进入模型。
   预测模型与真实多次波差一个源子波/尺度因子，由相减阶段吸收。

3. **分窗自适应相减** `subtract.adaptive_subtract`：每个 (炮 × 时间) 窗估计
   一个最小二乘匹配滤波器（含负 lag，容许时移/相位差；正规方程由移位视图
   精确构造，与滤波器施加方式严格对偶），相邻窗 50% 重叠、锥化加权叠加。

## 过度相减防护

`metrics.energy_leakage` 定义分窗能量泄漏率：

```
leak = max(0, E_removed − κ·E_model) / E_data
```

E_removed 为实际减掉的能量，E_model 为多次波模型在单标量最优拟合下可解释的
能量，κ 为波形整形余量。leak 显著大于 0 说明匹配滤波器在拟合一次波。
超过 `leak_thresh` 的窗口自动回退：先加大阻尼重估，仍超则退化为单标量拟合
（相减能量结构上不超过模型可解释能量）。

## 输出

`pipeline.run_srme` 返回的 `SRMEResult` 严格区分：

- `multiples_model`：原始预测多次波（未匹配）
- `multiples_matched`：匹配滤波后实际减去的多次波
- `primary`：保留信号（一次波估计），满足 primary + multiples_matched = regularized
- `leakage` / `diagnostics`：分窗泄漏率与 QC 指标

## 用法

```python
from geowave.multiple.pipeline import SRMEConfig, run_srme

cfg = SRMEConfig(dt_s=0.004, dx_m=25.0, n_iter=2, leak_thresh=0.2)
res = run_srme(cube, cfg, mask=observed_mask)   # cube: (n_shot, n_receiver, n_t)
```

完整演示（含与最优带通基线对比）：`python3 examples/srme_demo.py`
测试：`pytest tests/test_multiple_srme.py`

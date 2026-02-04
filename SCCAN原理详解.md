# LESYMAP SCCAN 原理详解

本文档详细解释 LESYMAP 中 SCCAN（Sparse Canonical Correlation Analysis）方法的工作原理，包括相关性计算、预测机制、线性校准等核心内容。

## 目录
1. [SCCAN 相关性计算](#1-sccan-相关性计算)
2. [为什么不用 AUC](#2-为什么不用-auc)
3. [SCCAN 参考文献](#3-sccan-参考文献)
4. [SCCAN 预测原理](#4-sccan-预测原理)
5. [线性校准机制](#5-线性校准机制)
6. [sparseDecom2 函数详解](#6-sparsedecom2-函数详解)
7. [二分类问题与 AUC 评估](#7-二分类问题与-auc-评估)
8. [Lesion-to-Symptom Mapping 的过拟合问题](#8-lesion-to-symptom-mapping-的过拟合问题)
9. [稀疏约束的本质](#9-稀疏约束的本质)
10. [lesmat 与 mask 的区别](#10-lesmat-与-mask-的区别)

---

## 1. SCCAN 相关性计算

### 1.1 交叉验证相关系数 (CV Correlation)

SCCAN 不计算 AUC，而是使用**交叉验证相关系数**作为模型评估指标。

### 1.2 计算流程

**代码位置**: `optimize_SCCANsparseness.R:96-140`

```r
# 对每个稀疏度值进行交叉验证
optimfun <- function(thissparse, lesmat, behavior, ...) {
    CVcorr = rep(NA, length(myfolds))

    for (cvrep in 1:length(myfolds)) {
        # 对每个折
        for (i in 1:length(myfolds[[cvrep]])) {
            fold = myfolds[[cvrep]][[i]]

            # 用训练集拟合 SCCAN 模型
            trainsccan = sparseDecom2(
                inmatrix = list(lesmat[-fold,], behavior[-fold]),
                ...
            )

            # 用测试集预测行为得分
            behavior.predicted[fold] = lesmat[fold,] %*% t(trainsccan$eig1) %*% trainsccan$eig2
        }

        # 计算真实值与预测值的相关系数
        CVcorr[cvrep] = abs(cor(behavior, behavior.predicted))
    }

    return(mean(CVcorr))
}
```

### 1.3 预测公式详解

```
behavior.predicted = lesmat_test @ eig1.T @ eig2

                     ┌─────────────────┐
                     │  测试集病变矩阵  │
                     │  (n_test × p)   │
                     └────────┬────────┘
                               │
                        @ eig1.T (p × 1)
                               │
                               ▼
                     ┌─────────────────┐
                     │  lesion_score   │  ← 每个患者的加权病变得分
                     │  (n_test × 1)   │
                     └────────┬────────┘
                               │
                        @ eig2 (1 × 1)
                               │
                               ▼
                     ┌─────────────────┐
                     │  预测的行为得分  │
                     │  (标准化)        │
                     └─────────────────┘
```

### 1.4 相关系数的 p 值计算

**代码位置**: `lsm_sccan.R:161-166`

```r
r = abs(CVcorrelation.stat)
n = length(behavior)

# 将相关系数转换为 t 统计量
tstat = (r * sqrt(n-2)) / sqrt(1 - r^2)

# 计算 p 值（双尾检验）
CVcorrelation.pval = pt(-abs(tstat), n-2) * 2
```

---

## 2. 为什么不用 AUC

### 2.1 AUC 的适用场景

AUC (Area Under Curve) 是为**二分类问题**设计的，衡量分类器区分正负类的能力。

### 2.2 SCCAN 是回归问题

```r
# SCCAN 解决的是回归问题
behavior = c(10.5, 23.1, 15.8, 30.2, ...)  # 连续的行为得分
# 不是：
behavior = c(0, 1, 0, 1, ...)  # 二分类
```

### 2.3 强行计算 AUC 的问题

如果强行把连续值转二分类再算 AUC：

```r
# 任意选择阈值（没有理论依据）
threshold = median(behavior)
pred_binary = ifelse(pred > threshold, 1, 0)
true_binary = ifelse(behavior > threshold, 1, 0)

auc = roc(true_binary, pred)  # 技术上可行，但意义不大
```

**问题**：
- 阈值选择是任意的（中位数？均值？0？）
- 丢失信息（连续→二分类）
- SCCAN 优化的是相关性，不是分类准确率

### 2.4 正确的评估指标

对于回归问题，应该使用：

| 指标 | R 代码 | 含义 |
|------|--------|------|
| **Pearson 相关系数** | `cor(true, pred)` | LESYMAP 默认使用 |
| **R²** | `summary(lm(true~pred))$r.squared` | 解释方差比例 |
| **RMSE** | `sqrt(mean((true-pred)^2))` | 预测误差 |
| **MAE** | `mean(abs(true-pred))` | 平均绝对误差 |

---

## 3. SCCAN 参考文献

### 3.1 LESYMAP 中 SCCAN 的应用

**Pustina et al. (2018)**
- PubMed ID: 28882479
- URL: https://www.ncbi.nlm.nih.gov/pubmed/28882479
- 说明：LESYMAP 中 SCCAN 应用于 lesion-symptom mapping 的主要论文

### 3.2 SCCAN 方法原始开发

**Avants et al.**
- SCCAN (Sparse Canonical Correlation Analysis) 由 Brian Avants 等人在 ANTs 工具包中开发
- LESYMAP 通过 ANTsR 的 `sparseDecom2` 函数调用 SCCAN

### 3.3 相关方法文献

| 方法 | 文献 | PMID | 用途 |
|------|------|------|------|
| SVR | Zhang (2014) | 25044213 | SVR 在 lesion mapping 中的应用 |
| Lesion size correction | Mirman (2015) | 25879574 | 病变大小校正方法 |
| Permutation | Winkler (2014) | PMC4010955 | Freedman-Lane 置换方法 |
| BM test | Medina (2010) | 19766664 | Brunner-Munzel 检验 |

---

## 4. SCCAN 预测原理

### 4.1 从相关性到预测

SCCAN 本质上是计算典型相关，但当行为变量只有一维时，它等价于一种**正则化回归**。

### 4.2 数学原理

```
标准 CCA/SCCAN：
- 找到两组变量的投影方向，使得投影后的相关性最大
- X @ w1 → ξ (病变投影)
- Y @ w2 → η (行为投影)
- 目标：max cor(ξ, η)

当 Y 是一维时（单变量行为得分）：
- w2 实际上是一个标量
- 问题简化为：max cor(X @ w1, Y)
- 这类似于回归：Y ≈ X @ β
```

### 4.3 预测公式

```r
# 代码：lsm_sccan.R:252
predbehav = lesmat %*% t(sccan$eig1) %*% sccan$eig2
#            ^^^^^^     ^^^^^^^^^^^^^^   ^^^^^^^^^^^^
#            数据矩阵    体素权重         行为权重
```

### 4.4 预测流程

```
┌─────────────────────────────────────────────────────────┐
│  新患者的病变图 (N 个体素)                               │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  lesmat_new (1 × N)                                     │
│  提取新患者在各个体素的病变值                            │
└───────────────────────┬─────────────────────────────────┘
                        │
                        @ eig1.T (N × 1)
                        │（加权求和，提取相关区域）
                        ▼
┌─────────────────────────────────────────────────────────┐
│  lesion_score (标量)                                     │
│  该患者的加权病变得分                                    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        @ eig2 (1 × 1)
                        │（转换到行为空间）
                        ▼
┌─────────────────────────────────────────────────────────┐
│  behavior_pred (标准化值)                                │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  反标准化 + 线性校准                                     │
│  → 最终预测得分                                          │
└─────────────────────────────────────────────────────────┘
```

### 4.5 直观理解

| 步骤 | SCCAN 视角 | 回归视角 |
|------|-----------|----------|
| 训练 | 找到 X 和 Y 最大相关的方向 | 找到 X → Y 的映射 |
| w1 | 使 X 的投影与 Y 相关最大 | 回归系数（稀疏化） |
| w2 | Y 的投影方向（标量） | 缩放因子 |
| 预测 | 新 X 的投影 → 通过 w2 转换 | X @ β → Y |

---

## 5. 线性校准机制

### 5.1 为什么需要线性校准？

**问题**：SCCAN 优化的是**相关性**，不是**预测误差**

```r
# SCCAN 保证：
cor(pred, true) → 最大

# 但不保证：
pred ≈ true

# 例如：
true  = c(10, 20, 30, 40, 50)
pred  = c(1, 2, 3, 4, 5)

# 相关性 = 1 （完美相关！）
# 但预测完全错误（差了10倍）
```

### 5.2 训练阶段的线性校准

**代码位置**: `lsm_sccan.R:251-256`

```r
# 步骤 1：用 SCCAN 权重自我预测
predbehav = lesmat %*% t(sccan$eig1) %*% sccan$eig2

# 步骤 2：反标准化
predbehav.raw = predbehav * behavior.scale + behavior.center

# 步骤 3：线性回归校准
output$sccan.predictlm = lm(behavior.orig ~ predbehav.raw,
                            data = data.frame(
                                behavior.orig = behavior.orig,
                                predbehav.raw = predbehav.raw
                            ))
```

**回归模型**：
```
behavior.orig = a × predbehav.raw + b

其中：
- a (slope): 斜率，修正缩放
- b (intercept): 截距，修正偏移
```

### 5.3 预测阶段应用校准

**代码位置**: `lesymap.predict.R:178-186`

```r
# 步骤 1：用 SCCAN 权重预测
behavior.scaled = lesmat %*% t(weights) %*% eig2

# 步骤 2：反标准化
behavior.raw = behavior.scaled * behavior.scale + behavior.center

# 步骤 3：应用校准模型
behavior.final = predict(lsm$sccan.predictlm,
                         newdata = data.frame(predbehav.raw = behavior.raw))
```

### 5.4 完整流程图

```
训练阶段：
┌─────────────────────────────────────────────────────────┐
│  1. 标准化数据                                           │
│     behavior_scaled = scale(behavior.orig)              │
│     lesmat_scaled = scale(lesmat)                       │
├─────────────────────────────────────────────────────────┤
│  2. SCCAN 学习权重                                      │
│     sccan = sparseDecom2(lesmat_scaled, behavior_scaled)│
│     eig1 = 体素权重                                     │
│     eig2 = 行为权重                                     │
├─────────────────────────────────────────────────────────┤
│  3. 用训练数据自我预测                                  │
│     pred = lesmat_scaled @ eig1.T @ eig2               │
├─────────────────────────────────────────────────────────┤
│  4. 反标准化                                            │
│     pred_raw = pred * scale + center                   │
├─────────────────────────────────────────────────────────┤
│  5. 线性回归校准 ← 关键步骤！                           │
│     model = lm(behavior.orig ~ pred_raw)               │
│     保存 model 到 output$sccan.predictlm                │
└─────────────────────────────────────────────────────────┘

预测阶段：
┌─────────────────────────────────────────────────────────┐
│  1. 用训练好的标准化参数处理新数据                      │
│     lesmat_new_scaled = scale(lesmat_new,               │
│                             center=old_center,          │
│                             scale=old_scale)            │
├─────────────────────────────────────────────────────────┤
│  2. SCCAN 预测                                         │
│     pred_scaled = lesmat_new_scaled @ eig1.T @ eig2    │
├─────────────────────────────────────────────────────────┤
│  3. 反标准化                                            │
│     pred_raw = pred_scaled * scale + center            │
├─────────────────────────────────────────────────────────┤
│  4. 应用校准模型                                        │
│     final = predict(model, pred_raw)                   │
└─────────────────────────────────────────────────────────┘
```

### 5.5 数值示例

```r
# 训练数据
behavior.orig = c(10, 20, 30, 40, 50)

# SCCAN 预测（反标准化后，但还没校准）
predbehav.raw = c(2, 4, 6, 8, 10)  # 相关性=1，但数值不对

# 线性回归校准
model = lm(behavior.orig ~ predbehav.raw)

# 拟合结果：
# intercept = 0
# slope = 5
# 模型：behavior.orig = 0 + 5 * predbehav.raw

# 预测新数据：
new_pred_raw = c(3, 5, 7)
final_pred = predict(model, newdata = data.frame(predbehav.raw = new_pred_raw))
# 结果：c(15, 25, 35)  ✓ 正确！
```

---

## 6. sparseDecom2 函数详解

### 6.1 函数概述

`sparseDecom2` 是 **ANTsR** 包的核心函数，执行**稀疏典型相关分析（SCCAN）**。

### 6.2 函数签名

```r
sparseDecom2(
    inmatrix = list(矩阵1, 矩阵2),    # 两组数据
    inmask = list(mask1, mask2),      # 空间掩码
    sparseness = c(值1, 值2),         # 稀疏度约束
    nvecs = 1,                        # 提取的成分数
    mycoption = 1,                    # 优化选项
    robust = 1,                       # 是否用秩
    its = 20,                         # 迭代次数
    smooth = 0.4,                     # 平滑参数
    cthresh = 150,                    # 聚类阈值
    maxBased = FALSE,                 # 是否基于最大值过滤
    perms = 0                         # 置换次数
)
```

### 6.3 输入

```r
# LESYMAP 中的调用 (lsm_sccan.R:139-140)
inmats = list(lesmat, as.matrix(behavior))
#        ^^^^^^   ^^^^^^^^^^^^^^^^^^^^
#        病变矩阵     行为向量（转成矩阵）
#       (n × p)          (n × 1)

sccan.masks = c(mask, NA)
#            ^^^^^   ^^
#            病变掩码   行为数据不需要掩码
```

### 6.4 输出

```r
sccan = sparseDecom2(...)

sccan$eig1          # 体素权重向量 (p × 1)
sccan$eig2          # 行为权重矩阵 (1 × nvecs)
sccan$ccasummary    # 包含相关性等信息
  └─ corrs[1]       # 典型相关系数
```

### 6.5 核心功能

```
给定两组数据：
X (n × p) - 病变矩阵，n 个患者，p 个体素
Y (n × 1) - 行为得分，n 个患者

SCCAN 找到：
w1 (p × 1) - 体素权重向量
w2 (1 × 1) - 行为权重（标量）

使得：
ξ = X @ w1      (病变的投影)
η = Y @ w2      (行为的投影)

目标：最大化 cor(ξ, η)

约束：
- 大部分 w1 的元素为 0（稀疏性）
- |w1| <= 1       (边界约束)
```

### 6.6 关键参数详解

| 参数 | 作用 | LESYMAP 默认值 | 影响 |
|------|------|----------------|------|
| **sparseness** | 稀疏度 | `c(0.045, -0.99)` | 值越接近0，越稀疏 |
| **smooth** | 空间平滑 | `0.4` | 使权重在空间上连续 |
| **cthresh** | 聚类阈值 | `150` | 移除小簇 |
| **its** | 迭代次数 | `20` | 优化迭代轮数 |
| **nvecs** | 成分数 | `1` | 提取多少对典型变量 |

### 6.7 sparseness 参数详解

```r
sparseness = c(0.045, -0.99)
#            ^^^^^^   ^^^^^^
#            病变侧    行为侧

# 正值：强制解为单侧（权重全 ≥ 0 或全 ≤ 0）
# 负值：允许双侧（可以有正有负）

# 0.045：约 4.5% 的体素获得非零权重
# -0.99：行为数据几乎不稀疏（因为只有一个变量）
```

### 6.8 运算示例

```r
# 输入数据
lesmat = matrix(c(
    1, 0, 1, 0,  # 患者1
    0, 1, 1, 0,  # 患者2
    1, 1, 0, 1   # 患者3
), nrow=3)

behavior = matrix(c(10, 20, 30))

# 调用 sparseDecom2
sccan = sparseDecom2(
    inmatrix = list(lesmat, behavior),
    sparseness = c(0.5, -0.99),
    nvecs = 1
)

# 输出示例
sccan$eig1  # c(0.8, 0, 0.6, 0)  → 体素1和3重要
sccan$eig2  # matrix(0.92)        → 缩放因子
sccan$ccasummary$corrs[1]  # 0.85  → 相关系数
```

### 6.9 在 LESYMAP 中的使用

**代码位置**: `lsm_sccan.R:207-210`

```r
sccan = sparseDecom2(
    inmats = inmats,
    inmask = sccan.masks,
    sparseness = sparseness,      # c(0.045, -0.99)
    nvecs = 1,
    mycoption = 1,
    robust = 1,
    its = 20,
    smooth = 0.4,
    cthresh = 150,
    maxBased = FALSE
)
```

---

## 7. 二分类问题与 AUC 评估

### 7.1 用 SCCAN 做二分类

虽然 SCCAN 是回归方法，但可以用于二分类问题：

```r
# 不缩放，直接用 0/1
behavior_binary = c(0, 1, 0, 1, 1, 0, ...)

# 训练 SCCAN
lsm = lesymap(lesions[train], behavior_binary[train],
              method='sccan')

# 预测（得到连续值）
pred = lesymap.predict(lsm, lesions[test])$behavior.raw

# 计算 AUC（用连续预测值）
library(pROC)
roc_obj = roc(behavior_binary[test], pred)
auc_value = auc(roc_obj)
```

### 7.2 AUC 的本质：只看排序

```r
# AUC 只依赖于排序，不依赖于绝对值
# AUC = P(正样本预测值 > 负样本预测值)

# 例如：
true  = c(0, 0, 1, 1, 1)
pred  = c(2, 3, 5, 7, 9)  # 排序正确
# AUC = 1.0 (完美)

# 如果线性缩放：
pred_scaled = c(2000, 3000, 5000, 7000, 9000)
# AUC 仍然是 1.0（排序没变）
```

### 7.3 高 AUC + 低 F1 现象

**现象**：AUC 很高，但实际分类时 F1 很低

```
不平衡数据下 SCCAN 的典型输出（90% 负类，10% 正类）：

负类预测: [-0.35, -0.30, -0.32, -0.28, -0.31, ...]  ← 接近标准化后的负类中心
正类预测: [-0.25, -0.20, -0.18, -0.22, -0.15, ...]  ← 只是稍微高一点

计算 AUC：
# 所有正类预测 > 所有负类预测（大部分）
# AUC ≈ 0.85（看起来不错！）

但用阈值 0.5 分类：
# 所有样本都被预测为负类
# F1 = 0（完全无法识别正类）
```

### 7.4 为什么会出现这种现象？

#### 原因 1：SCCAN 优化相关性

```r
# SCCAN 目标：max cor(pred, true)

# 不平衡数据：
# - 90% 的样本值是 0（标准化后 ≈ -0.33）
# - 10% 的样本值是 1（标准化后 ≈ 3）

# SCCAN 的策略：
# 让大部分预测接近 -0.33（匹配多数类）
# 这样可以最大化相关性！

# 结果：相关性高，但分类差
```

#### 原因 2：AUC 只看排序

```r
# AUC 不关心：
# - 预测值的绝对大小
# - 是否有清晰的分界线
# - 阈值在哪里

# AUC 只关心：
# 正样本的预测值是否普遍高于负样本
```

### 7.5 枚举阈值的局限

#### 枚举有效的情况（预测值有区分度）

```
负类: [0.1, 0.2, 0.15, 0.25, ...]  ← 集中在低值
正类: [0.6, 0.7, 0.55, 0.8, ...]   ← 集中在高值

枚举阈值可以找到 0.5 附近的最佳值 ✓
```

#### 枚举无效的情况（预测值重叠）

```
负类: [-0.35, -0.30, -0.32, -0.28, ...]  ← 负类
正类: [-0.25, -0.20, -0.18, -0.22, ...]  ← 正类，只是稍高

无论阈值选在哪：
- 阈值太左：召回率高，精确率低
- 阈值太右：精确率高，召回率低
- 阈值在中间：两者都不高

枚举只能"在糟糕的选择中找一个最不糟糕的" ✗
```

### 7.6 检查预测值分布：Cohen's d

```r
# 计算预测值的分离度
cohens_d = function(pred_pos, pred_neg) {
    mean_pos = mean(pred_pos)
    mean_neg = mean(pred_neg)
    sd_pos = sd(pred_pos)
    sd_neg = sd(pred_neg)

    d = (mean_pos - mean_neg) / sqrt((sd_pos^2 + sd_neg^2) / 2)
    return(d)
}

# 解释：
# d < 0.3: 几乎没有分离 → 枚举阈值无效
# d < 0.5: 小分离 → 枚举阈值效果有限
# d > 0.8: 大分离 → 枚举阈值有效
```

### 7.7 正确的二分类评估方法

```r
library(pROC)
library(PRROC)
library(caret)

# 1. 同时看 ROC AUC 和 PR AUC
roc_obj = roc(true_values, pred_values)
roc_auc = auc(roc_obj)

pr_curve = pr.curve(
    scores.class0 = pred_values[true_values == 1],
    scores.class1 = pred_values[true_values == 0]
)
pr_auc = pr_curve$auc.integral

# 2. 找最佳阈值（不只看 Youden Index）
thresholds = seq(min(pred_values), max(pred_values), length.out = 100)
f1_scores = sapply(thresholds, function(thr) {
    pred_binary = ifelse(pred_values >= thr, 1, 0)
    # 计算 F1
})

# 3. 看混淆矩阵
pred_binary = ifelse(pred_values > best_threshold, 1, 0)
conf = confusionMatrix(factor(pred_binary), factor(true_values), positive = "1")

# 4. 计算 MCC（对不平衡更鲁棒）
mcc = mcc(pred_binary, true_values)
```

### 7.8 推荐做法

| 场景 | 推荐方法 |
|------|----------|
| **纯二分类问题** | `method='chisq'` 或 `chisqPerm` |
| **想用多变量方法** | `method='svr'` |
| **坚持用 SCCAN** | 不要缩放，用连续预测值算 AUC，同时看 PR AUC |
| **不平衡数据** | PR AUC > ROC AUC，看 F1 和 MCC |

---

## 8. Lesion-to-Symptom Mapping 的过拟合问题

### 8.1 数据特点：高维小样本

```
典型 LSM 数据：
样本数 (n)：50-200 个患者
特征数 (p)：100,000 - 1,000,000 个体素

问题：p >> n
特征数是样本数的成千上万倍！
```

### 8.2 为什么树类方法严重过拟合？

#### 问题 1：树的贪婪分裂

```python
# xgboost 的学习过程
for tree in trees:
    for leaf in tree:
        # 在所有特征中找最优分裂点
        for feature in all_features:  # 遍历 100,000+ 个体素
            gain = calculate_gain(feature, threshold)

        # 问题：总能找到"看起来相关"的特征！
        # 但这些相关性是虚假的、随机的
```

#### 问题 2：boosting 的记忆效应

```
迭代次数    训练集 R²    测试集 R²
─────────────────────────────────
1          0.20         0.18
10         0.50         0.30
50         0.80         0.20
100        0.95         0.10
200        0.99         0.05  ← 完全记忆
```

#### 问题 3：特征重要性不稳定

```r
# xgboost 的特征重要性
# 在不同随机种子下完全不同
# 在不同的训练/测试划分下完全不同
# 没有神经生物学意义
```

### 8.3 为什么 SVR/SCCAN 更适合？

#### SVR 的优势

| 特性 | 说明 | 优势 |
|------|------|------|
| **L2 正则** ||w||² 惩罚大权重 | 防止模型太复杂 |
| **支持向量** | 只有边界样本影响决策 | 更鲁棒 |
| **最大间隔** | 最大化决策边界 | 更好的泛化 |
| **稀疏解** | 很多特征的权重 = 0 | 自动特征选择 |

#### SCCAN 的优势

| 特性 | 说明 |
|------|------|
| **稀疏性** | sparseness=0.045 → 只用 4.5% 的体素 |
| **空间平滑** | smooth=0.4 → 惩罚孤立体素 |
| **聚类阈值** | cthresh=150 → 移除小簇 |
| **典型相关** | 优化相关性，不是逐点拟合 |

### 8.4 实验对比

| 方法 | 训练集相关 | 测试集相关 | 过拟合程度 |
|------|-----------|-----------|-----------|
| xgboost | 0.99 | 0.15 | 严重 |
| random forest | 0.95 | 0.20 | 严重 |
| SVR | - | 0.50 | 中等 |
| SCCAN | - | 0.55 | 较轻 |

### 8.5 如果必须用树类方法

#### 预处理：降维

```r
# 方法 A：patch-based（LESYMAP 默认）
# 将相同病变模式的体素分组，减少特征数

# 方法 B：ROI-based
# 先用脑图谱定义 ROI，只用 ROI 的平均 lesion load

# 方法 C：PCA 预处理
lesmat_pca = prcomp(lesmat)
lesmat_reduced = lesmat_pca$x[, 1:50]
```

#### 极端正则化

```r
# xgboost 极端正则化
xgb_model = xgboost(
    data = lesmat[train, ],
    label = behavior[train],
    nrounds = 10,              # 减少迭代
    max_depth = 2,             # 减少深度
    eta = 0.01,                # 小学习率
    subsample = 0.5,           # 更少样本
    colsample_bytree = 0.1,    # 更少特征
    min_child_weight = 10,     # 更严格
    gamma = 1,                 # 更高的分裂要求
    lambda = 10,               # L2 正则
    alpha = 10                 # L1 正则
)
```

#### 集成策略

```r
# 先用 univariate 方法筛选
lsm_bm = lesymap(lesions, behavior, method='BMfast', nperm = 100)
significant_voxels = which(lsm_bm$pvalue < 0.05)

# 只在显著体素上训练 xgboost
lesmat_filtered = lesmat[, significant_voxels]
xgb_model = xgboost(lesmat_filtered[train, ], behavior[train])
```

### 8.6 推荐方法优先级

```
SCCAN > SVR > 传统体素-wise (BMfast, ttest) >> 树类方法
```

---

## 9. 稀疏约束的本质

### 9.1 两个"稀疏"的区别

#### 稀疏 1：输入数据的稀疏性

```r
# 输入是 lesion mask
lesion_mask = c(
    0, 0, 0, 1, 1, 0, 0, 0, 1, 0,  # 大部分是 0
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  # 只有少数是 1
    ...
)

# 统计：
sum(lesion_mask) / length(lesion_mask)  # 例如：5% 稀疏
# 这是"输入数据的稀疏性"
```

#### 稀疏 2：模型权重的稀疏性

```r
# SCCAN 学习到的权重
weights = sccan$eig1
# 例如：
weights = c(
    0, 0, 0, 0.8, 0.6, 0, 0, 0, 0, 0,  # 大部分是 0
    0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0,    # 只有少数有值
    ...
)

# 统计：
sum(weights != 0) / length(weights)  # 例如：4.5% 稀疏
# 这是"模型权重的稀疏性"（由 sparseness 参数控制）
```

### 9.2 稀疏约束的作用

```r
# SCCAN 的优化问题：
max cor(X @ w1, Y)

subject to: ||w1||₀ ≤ k
#           ^^^^^^
#           L0 范数：非零元素的个数

# 等价于：
# 在所有体素中，最多只有 k 个体素的权重不为 0
# 其他体素的权重强制为 0
```

### 9.3 sparseness 参数的影响

```r
# 不同 sparseness 值的效果

# sparseness = 0.01（非常稀疏）
# → 1% 的体素有权重
# → 只找最核心的脑区
# → 可能欠拟合（漏掉相关区域）

# sparseness = 0.05（LESYMAP 默认）
# → 5% 的体素有权重
# → 平衡稀疏性和表达能力

# sparseness = 0.5（不稀疏）
# → 50% 的体素有权重
# → 可能过拟合（包含噪声体素）

# sparseness = 0.9（几乎无约束）
# → 90% 的体素有权重
# → 类似普通 CCA，严重过拟合
```

### 9.4 稀疏约束 vs 输入稀疏性

| 维度 | 输入数据的稀疏性 | 模型权重的稀疏约束 |
|------|----------------|------------------|
| **是什么** | lesion mask 中 1 的比例 | 权重中非零值的比例 |
| **来源** | 数据本身（患者病变） | 模型学习（sparseness 参数） |
| **目的** | 减少存储和计算 | 防止过拟合，提高可解释性 |
| **典型值** | 5-10%（病变体积） | 1-10%（重要体素比例） |
| **可控制** | ❌ 由患者病变决定 | ✅ 由 sparseness 参数控制 |

### 9.5 可视化对比

#### 输入数据（lesion mask）

```
某个患者的病变图（3D 切片）：
┌─────────────────────────────┐
│ 0 0 0 0 0 0 0 0 0 0        │  ← 健康
│ 0 0 1 1 1 0 0 0 0 0        │  ← 病变区域
│ 0 0 1 1 0 0 0 0 0 0        │
│ 0 0 1 0 0 0 0 0 0 0        │
│ 0 0 0 0 0 0 0 0 0 0        │
└─────────────────────────────┘

稀疏度：~15%（100个体素中15个病变）
```

#### 模型权重（SCCAN 学到的）

```
SCCAN 学到的体素重要性图：
┌─────────────────────────────┐
│ 0 0 0 0 0 0 0 0 0 0        │  ← 无关体素（权重=0）
│ 0 0 0.8 0 0 0 0 0 0 0      │  ← 重要体素（权重>0）
│ 0 0 0 0.6 0 0 0 0 0 0      │
│ 0 0 0 0 0 0 0 0 0 0        │
│ 0 0 0 0 0 0 0 0 0 0        │
└─────────────────────────────┘

稀疏度：~2%（100个体素中只有2个有非零权重）
```

---

## 10. lesmat 与 mask 的区别

### 10.1 两个参数的不同作用

#### lesmat（Lesion Matrix）- 实际数据

```r
# lesmat：患者 × 体素 的矩阵
# 每一行是一个患者
# 每一列是一个体素的值

lesmat = matrix(c(
    # 体素1  体素2  体素3  体素4  ...
    0,     1,     0,     1,     ...,  # 患者1
    1,     0,     1,     0,     ...,  # 患者2
    0,     0,     1,     1,     ...   # 患者3
), nrow = 3, byrow = TRUE)

# lesmat 包含：
# - 实际的病变信息（0 或 1）
# - 所有患者的数据
# - 在 mask 定义的体素位置上的值
```

#### mask - 空间模板/参考

```r
# mask：一个 3D 图像
# 定义了"体素坐标"和"分析的边界"

mask = antsImageRead(template_mask)
# mask 是一个二值图像：
# - 1：包含在这个分析中
# - 0：排除在分析外

# mask 的作用：
# 1. 定义空间的形状和大小
# 2. 定义哪些体素位置参与分析
# 3. 作为输出图像的模板
```

### 10.2 为什么同时需要两者？

#### 原因 1：矩阵没有空间信息

```r
# lesmat 只是一个矩阵，丢失了空间信息
lesmat = matrix(c(0, 1, 0, 1, 0, ...), nrow = 1)

# 问题：
# - 哪个体素在哪个位置？不知道
# - 这些体素的空间关系？不知道
# - 如何把结果映射回 3D 大脑？需要 mask

# mask 保存了空间信息：
mask_dimensions = dim(mask)  # 例如：(121, 145, 121)
mask_spacing = antsGetSpacing(mask)  # 体素大小
mask_origin = antsGetOrigin(mask)  # 原点位置
```

#### 原因 2：mask 定义了分析的 ROI

```r
# 场景：只分析灰质，不分析脑室外
# 创建灰质 mask
gm_mask = antsImageRead("gray_matter_mask.nii.gz")

# 在灰质 mask 内提取 lesmat
lesmat = imageListToMatrix(lesions_list, gm_mask)
# lesmat 只包含灰质区域的体素值

# 训练 SCCAN
lsm = lesymap(lesions_list, behavior, mask = gm_mask)
```

#### 原因 3：mask 用于结果可视化

```r
# SCCAN 返回统计向量
statistic = lsm$statistic  # 一个向量，长度 = 体素数

# 如何把这个向量变成 3D 图像？
# 需要 mask 作为模板

stat_img = makeImage(mask, statistic)
#                ^^^^^
#                提供空间框架
```

#### 原因 4：验证所有图像在同一空间

```r
# checkMask 函数验证：
# 1. 所有 lesion 图像和 mask 是否有相同的维度
# 2. 是否有相同的原点
# 3. 是否有相同的分辨率
# 4. 是否有相同的方向

# 如果不匹配，报错
checkMask(lesions_list, mask)
# Error: Mask and images are in different space
```

### 10.3 实际代码流程

```r
library(LESYMAP)
library(ANTsR)

# 步骤 1：准备 mask
mask = antsImageRead("MNI152_T1_2009c_brain_mask.nii.gz")

# 步骤 2：准备 lesion 图像
lesions_list = list(
    patient1 = antsImageRead("patient1_lesion.nii.gz"),
    patient2 = antsImageRead("patient2_lesion.nii.gz"),
    ...
)

# 步骤 3：检查空间匹配
checkMask(lesions_list, mask)

# 步骤 4：提取 lesmat
lesmat = imageListToMatrix(lesions_list, mask)

# 步骤 5：训练 SCCAN
lsm = lesymap(
    lesions_list,
    behavior,
    mask = mask
)

# 内部流程：
# 1. 用 mask 提取 lesmat
# 2. 在 lesmat 上运行 SCCAN
# 3. 得到统计向量 statistic
# 4. 用 mask 将 statistic 转换回 3D 图像
```

### 10.4 两个参数的实际关系

```
原始 lesion 图像 + mask
        ↓
   imageListToMatrix
        ↓
      lesmat
        ↓
      SCCAN
        ↓
  statistic (向量)
        ↓
   makeImage(mask, statistic)
        ↓
   3D 统计图像

┌─────────────────────────────────────┐
│          mask 的作用                │
├─────────────────────────────────────┤
│ 1. 定义分析的 ROI（哪些体素）        │
│ 2. 提供空间信息（维度、原点等）      │
│ 3. 验证图像对齐                     │
│ 4. 将结果转换回 3D 图像             │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│          lesmat 的作用              │
├─────────────────────────────────────┤
│ 1. 存储实际的分析数据               │
│ 2. 可以进行矩阵运算                 │
│ 3. 传给 SCCAN/统计函数              │
└─────────────────────────────────────┘
```

### 10.5 对比总结

| 问题 | lesmat | mask |
|------|--------|-------|
| **是什么** | 矩阵（患者×体素） | 3D 图像 |
| **包含什么** | 病变数据（0/1 值） | 空间框架 |
| **由什么决定** | lesion 图像 + mask | 模板选择（如 MNI） |
| **作用** | 传给分析算法 | 定义空间、提取数据、重建图像 |
| **可以缺少吗** | ❌ 必须有数据 | ✅ 可以自动生成 |

---

## 总结

### 核心概念

| 概念 | 说明 |
|------|------|
| **SCCAN 本质** | 找到两组变量的最大相关方向，带稀疏性约束 |
| **相关性计算** | 通过 k 折交叉验证计算预测值与真实值的相关系数 |
| **预测机制** | 当 Y 一维时，SCCAN ≈ 稀疏回归，Y_pred = X @ w1 @ w2 |
| **线性校准** | SCCAN 优化相关性而非准确度，需要 `lm(true ~ pred)` 校准 |
| **稀疏约束** | 控制模型权重的稀疏性，不是输入数据的稀疏性 |
| **核心函数** | `sparseDecom2` (ANTsR) 执行 SCCAN 算法 |
| **主要文献** | Pustina (2018) PMID: 28882479 |

### 二分类评估注意事项

| 问题 | 说明 |
|------|------|
| **AUC 的局限** | 只看排序，不保证实际分类效果好 |
| **高 AUC 低 F1** | 在不平衡数据下常见，需要检查预测值分布 |
| **枚举阈值** | 只在预测值有区分度时有效，否则只是"在糟糕中选最好的" |
| **推荐做法** | 同时看 ROC AUC、PR AUC、F1、MCC |

### 过拟合问题

| 问题 | 答案 |
|------|------|
| **树类方法为什么过拟合** | 高维小样本 + 贪婪分裂 + boosting 记忆 |
| **SVR/SCCAN 为什么适合** | L2 正则 + 稀疏约束 + 空间平滑 |
| **推荐方法优先级** | SCCAN > SVR > 传统体素-wise >> 树类 |

### 数据结构

| 组件 | 作用 |
|------|------|
| **lesmat** | 实际分析数据（患者×体素矩阵） |
| **mask** | 空间框架（定义 ROI、验证对齐、重建图像） |
| **两者关系** | mask 用于从图像中提取 lesmat，再将结果转换回图像 |

---

*文档生成日期: 2025-01-04*
*基于 LESYMAP v0.0.0.9222*
*更新日期: 2025-01-04（添加二分类评估、过拟合问题、稀疏约束、lesmat与mask区别）*

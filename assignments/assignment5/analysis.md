# ARAP 参数化解析

## ARAP 能量
ARAP 算法通过最小化局部形变能量来实现尽可能保持刚性的平面映射。常规的 ARAP 能量形式为  
$$
E(u,L) = \sum_{t=1}^T{\sum_{i=0}^2\cot{\theta_i^t}\|(u_i^t-u_{i+1}^t) - L_t(x_i^t-x_{i+1}^t)\|^2}
$$
其中 $u$ 为参数化坐标，$L_t$ 为三角形 $t$ 的局部正交变换，$x$是每个三角形刚体地移动到平面后的坐标

## 1. 初始映射
- 提前获取一个初始平面嵌入（如作业 4 中的Tutte参数化结果）。
- 该初始映射用于在后续迭代时提供起始坐标。
在完成作业 4（Tutte Parameterization）后，可以在 `void arap(...)` 中将第一个输入网格作为原始 3D 网格，第二个输入网格则作为初始平面坐标。具体初始化片段示例：
```cpp
// ...existing code...
auto halfedge_mesh = operand_to_openmesh(&input);
auto iter_mesh = operand_to_openmesh(&iters);
// ...existing code...
// 准备 iter_mesh 作为局部和全局阶段迭代使用的2D坐标初值
```

## 2. 局部步骤（Local Phase）
在局部步骤中，我们固定$u$，计算每个三角形的局部正交变换$L_t$。

首先需要计算每个三角形的雅可比矩阵$J_t$

$$
J_t(u) = \sum_{i=0}^{2} \cot{\theta_i^t}(u_i^t - u_j^t) (x_i^t - x_j^t)^T
$$

随后做带符号的SVD分解

$$
J_t(u) = U \Sigma V^T
$$

其中$U$、$V$为正交矩阵，$\Sigma$为对角矩阵$diag(\sigma_1, \sigma_2)$

即得到了局部正交变换$L_t$
$$
L_t = U V^T
$$

- 对每个三角形分别计算其局部正交变换 Lt，用以近似三角形从 3D 到 2D 的形变。
- 在代码中，通过对雅可比矩阵 Jacobi 做 SVD 分解得到正交矩阵，若出现翻转则需进行相应的修正。
在每次迭代的局部阶段，会计算每个面对应的局部正交变换 Lt。这里使用奇异值分解 (SVD) 来逼近刚性旋转矩阵：
```cpp
// ...existing code...
Eigen::JacobiSVD<Eigen::MatrixXd> svd(
    Jacobi[face_idx],
    Eigen::DecompositionOptions::ComputeThinU |
        Eigen::DecompositionOptions::ComputeThinV);
// ...existing code...
Jacobi[face_idx] = svd_u * S * svd_v.transpose();
// ...existing code...
```
若出现翻转 (det(Jacobi[face_idx]) < 0)，则通过修正奇异值矩阵 S 中的符号位来解决。处理完后，便得到了每个三角形局部可逆的刚性近似变换。

## 3. 全局步骤（Global Phase）
在全局步骤中，我们固定每个三角形的局部正交变换$L_t$，更新所有顶点的坐标$u$。

通过对ARAP能量求导，我们得到

$$
\sum_{j\in N(i)} (\cot{\theta_{ij}}+\cot{\theta_{ji}})(u_i - u_j) = \sum_{j \in N(i)} (\cot{\theta_{ij}}L_{t(i,j)}+\cot{\theta_{ji}}L_{t(j,i)}) (x_i - x_j)
$$

- 固定各三角形的局部正交变换后，构造并求解全局稀疏线性方程组，以更新所有顶点在平面中的坐标。
- 稀疏矩阵的系数可在迭代前一次性预分解，提高效率。
在全局阶段，需要根据局部 Lt 重新求解所有顶点坐标。可以在代码里看到“矩阵预分解”与构造向量 bx、by 的流程：
```cpp
// ...existing code...
bx(vertex_idx[i]) += b(0, i);
by(vertex_idx[i]) += b(1, i);
// ...existing code...
```
然后调用
```cpp
Eigen::VectorXd ux = solver.solve(bx);
Eigen::VectorXd uy = solver.solve(by);
```
把解得到的 ux、uy 分别填回新的平面坐标。

## 4. 迭代
- 重复局部与全局步骤，直到收敛或达到最大迭代次数。
- 每次迭代会逐步减小网格的局部形变误差，使三角形在目标平面上更加接近原始网格形态。
最后通过一个 do-while 循环多次执行局部与全局步骤，直到收敛。核心逻辑如下：
```cpp
// ...existing code...
do {
    // 局部步骤：SVD 计算Jacobi[face_idx]
    // 全局步骤：solver.solve( bx ), solver.solve( by )
    // 计算误差 err
} while (now_iter < max_iter && abs(err - err_pre) > 1e-7);
```
随着迭代进行，每一面局部刚性近似与全局坐标同步调整，能量逐步下降，网格在平面上的整体形变就越来越接近“保形”。

## 5. 代码简要分析
- 函数 arap() 完成主要流程，包括：
  - 初始化：读取网格顶点数量、面片信息及余切权重 (cotangents)。
  - 预处理：构造并分解线性系统，用于全局阶段的快速求解。
  - 局部相似变换计算：利用 SVD 得到正交矩阵，解决翻转并逼近刚性变换。
  - 全局求解：通过稀疏矩阵求解更新所有顶点的 (x,y)。
  - 误差计算：判断收敛条件并控制迭代次数。
在 `node_arap_parameterization.cpp` 中还能看到一些预先算好的 b_pre、Jacobi_pre 用于加速每次局部-全局循环。

通过这一套 Local-Global 迭代方案，ARAP 能在平面中最大程度保留原始网格的局部几何特征，从而获得更形变最小的参数化效果。

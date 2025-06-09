# QEM网格简化

## 算法流程

1. 计算每个顶点的$Q$矩阵。

2. 预处理所有的合法点对，包括相邻顶点和距离非常近的不相邻顶点（可选），求出其收缩后的新顶点位置和收缩损失，存入以收缩损失排序的优先队列中。

3. 迭代以下操作，直到模型的面数小于目标面数：

    1. 从优先队列中取出损失最小的点对$(v_1,v_2)$，进行收缩操作，记新顶点为$v_{new}$。

    2. 计算新顶点$v_{new}$的$Q$矩阵，$Q_{new}=Q_1+Q_2$。

    3. 将优先队列中所有与$v_1$或$v_2$关联的点对$(v_1,u)$和$(v_2,u)$更新为$(v_{new},u)$，并重新计算新顶点位置和收缩损失，填入优先队列中。

## 代码实现

### 辅助数据结构

```cpp
struct VertexPair {
    // 记录点对的顶点句柄
    MyMesh::VertexHandle v1;
    MyMesh::VertexHandle v2;
    // 通过记录点对的版本号，避免对失效/移动过的顶点进行收缩
    int v1_version;
    int v2_version;
    // 收缩后的新顶点位置
    Eigen::Vector3f optimal_point;
    // 收缩损失
    float cost;

    // 用于优先队列排序
    bool operator>(const VertexPair& other) const
    {
        return cost > other.cost;
    }
};
```

### 计算所有顶点的$Q$矩阵

$$
\Delta(v) = v^TQv = v^T \left(\sum_{p\in planes(v)}K_p\right)v
$$

其中

$$
K_p = pp^T = \begin{bmatrix}
    a^2 & ab & ac & ad \\
    ab & b^2 & bc & bd \\
    ac & bc & c^2 & cd \\
    ad & bd & cd & d^2
\end{bmatrix}
$$


```cpp
    // 计算顶点对应的Q矩阵
    // 为每个顶点添加一个属性vprop_Q，用于存储Q矩阵
    OpenMesh::VPropHandleT<Eigen::Matrix4f> vprop_Q;
    halfedge_mesh->add_property(vprop_Q);
    for (auto v_it = halfedge_mesh->vertices_begin();
         v_it != halfedge_mesh->vertices_end();
         ++v_it) {
        auto vh = *v_it;
        Eigen::Matrix4f Q = Eigen::Matrix4f::Zero();
        for (auto fv_it = halfedge_mesh->vf_begin(vh);
             fv_it != halfedge_mesh->vf_end(vh);
             ++fv_it) {
            auto fh = *fv_it;
            // plane_eq是fh的平面方程的系数(a, b, c, d)
            Eigen::Vector4f plane_eq =
                compute_plane_equation(halfedge_mesh, fh);
            // 计算K_p = pp^T
            Eigen::Matrix4f Q_i = plane_eq * plane_eq.transpose();
            // 计算Q=\sum(K_p)
            Q += Q_i;
        }
        halfedge_mesh->property(vprop_Q, vh) = Q;
    }
```

### 初始化点对优先队列

1. 遍历所有点对（参考代码中仅遍历所有边），记录：
    - 点对的顶点句柄
    - 点对的版本号
2. 根据顶点的坐标和$Q$矩阵之和，计算最优收缩位置$\bar{v}$
    - 收缩损失：$\Delta(\bar(v))=\bar{v}^T Q \bar{v}$
    - 令收缩损失最小：$\frac{\partial \Delta(\bar{v})}{\partial x} = \frac{\partial \Delta(\bar{v})}{\partial y} = \frac{\partial \Delta(\bar{v})}{\partial z} = 0$
    - 等价于：
        $$
        \begin{bmatrix}
        q_{11} & q_{12} & q_{13}  \\
        q_{21} & q_{22} & q_{23} \\
        q_{31} & q_{32} & q_{33} 
        \end{bmatrix}\bar{v} = -\begin{bmatrix}
        q_{14} \\
        q_{24} \\
        q_{34}
        \end{bmatrix}
        $$
    - 若求解出的位置不合理，选择两端点和中点中收缩代价最小的点作为新顶点位置

这里给出创建点对的函数

```cpp
VertexPair create_vertex_pair(
    std::shared_ptr<MyMesh> halfedge_mesh,
    OpenMesh::VPropHandleT<Eigen::Matrix4f> vprop_Q,
    OpenMesh::VPropHandleT<int> vprop_version,
    MyMesh::VertexHandle vh1,
    MyMesh::VertexHandle vh2)
{
    VertexPair pair;

    if (vh1.idx() < vh2.idx()) {
        pair.v1 = vh1;
        pair.v2 = vh2;
    }
    else {
        pair.v1 = vh2;
        pair.v2 = vh1;
    }

    pair.v1_version = halfedge_mesh->property(vprop_version, pair.v1);
    pair.v2_version = halfedge_mesh->property(vprop_version, pair.v2);

    Eigen::Matrix4f Q1 = halfedge_mesh->property(vprop_Q, pair.v1);
    Eigen::Matrix4f Q2 = halfedge_mesh->property(vprop_Q, pair.v2);

    Eigen::Matrix4f Q_sum = Q1 + Q2;

    Eigen::Matrix3f A = Q_sum.block<3, 3>(0, 0);
    Eigen::Vector3f b = Q_sum.block<3, 1>(0, 3);

    Eigen::Vector3f optimal_point;
    bool use_optimal = false;
    if (std::abs(A.determinant()) > 1e-8) {
        Eigen::LDLT<Eigen::Matrix3f> solver(A);
        Eigen::Vector3f x = solver.solve(-b);
        // 检查新点是否离原边太远
        auto p1 = halfedge_mesh->point(pair.v1);
        auto p2 = halfedge_mesh->point(pair.v2);
        Eigen::Vector3f v1(p1[0], p1[1], p1[2]);
        Eigen::Vector3f v2(p2[0], p2[1], p2[2]);
        float edge_len = (v1 - v2).norm();
        float dist1 = (x - v1).norm();
        float dist2 = (x - v2).norm();

        if (dist1 < 5 * edge_len && dist2 < 5 * edge_len) {
            optimal_point = x;
            use_optimal = true;
        }
    }
    if (!use_optimal) {
        // 选择端点或中点
        auto p1 = halfedge_mesh->point(pair.v1);
        auto p2 = halfedge_mesh->point(pair.v2);
        Eigen::Vector3f v1(p1[0], p1[1], p1[2]);
        Eigen::Vector3f v2(p2[0], p2[1], p2[2]);
        auto mid = (v1 + v2) / 2.0f;
        float cost_1 = compute_cost(Q_sum, v1);
        float cost_2 = compute_cost(Q_sum, v2);
        float cost_mid = compute_cost(Q_sum, mid);
        if (cost_1 < cost_2 && cost_1 < cost_mid) {
            optimal_point = v1;
        }
        else if (cost_2 < cost_1 && cost_2 < cost_mid) {
            optimal_point = v2;
        }
        else {
            optimal_point = mid;
        }
    }
    pair.optimal_point = optimal_point;
    pair.cost = compute_cost(Q_sum, pair.optimal_point);
    return pair;
}
```

利用以上函数，初始化优先队列

```cpp
    // 初始化优先队列
    std::priority_queue<
        VertexPair,
        std::vector<VertexPair>,
        std::greater<VertexPair>>
        heap_vertex_pairs;

    // 创建初始顶点对
    for (auto e_it = halfedge_mesh->edges_begin();
         e_it != halfedge_mesh->edges_end();
         ++e_it) {
        auto eh = *e_it;
        auto vh1 = eh.v0();
        auto vh2 = eh.v1();

        VertexPair pair =
            create_vertex_pair(halfedge_mesh, vprop_Q, vprop_version, vh1, vh2);

        heap_vertex_pairs.push(pair);
    }
```

### 迭代

```cpp
    // 迭代收缩顶点对
    size_t target_num_faces =
        static_cast<size_t>(simplification_ratio * halfedge_mesh->n_faces());
    size_t current_num_faces = halfedge_mesh->n_faces();

    do {
        VertexPair pair = heap_vertex_pairs.top();
        heap_vertex_pairs.pop();
        auto vh1 = pair.v1;
        auto vh2 = pair.v2;
        auto optimal_point = pair.optimal_point;

        // 检查顶点是否被移动过（版本号是否一致）
        if (pair.v1_version != halfedge_mesh->property(vprop_version, vh1) ||
            pair.v2_version != halfedge_mesh->property(vprop_version, vh2)) {
            continue;
        }

        // 检查顶点是否被删除
        if (!halfedge_mesh->is_valid_handle(vh1) ||
            !halfedge_mesh->is_valid_handle(vh2)) {
            continue;
        }

        // 找到要收缩的边
        bool collapse_success = false;
        for (auto heh : halfedge_mesh->voh_range(vh1)) {
            if (halfedge_mesh->to_vertex_handle(heh) == vh2) {
                if (halfedge_mesh->is_valid_handle(heh) &&
                    halfedge_mesh->is_collapse_ok(heh)) {
                    halfedge_mesh->set_point(
                        vh2,
                        MyMesh::Point(
                            optimal_point.x(),
                            optimal_point.y(),
                            optimal_point.z()));
                    // 收缩边，OpenMesh会删去vh1，保留vh2
                    halfedge_mesh->collapse(heh);
                    halfedge_mesh->property(vprop_version, vh2)++;
                    collapse_success = true;
                }
                break;
            }
        }
        if (!collapse_success) {
            continue;
        }

        // 更新Q矩阵
        Eigen::Matrix4f Q1 = halfedge_mesh->property(vprop_Q, vh1);
        Eigen::Matrix4f Q2 = halfedge_mesh->property(vprop_Q, vh2);
        halfedge_mesh->property(vprop_Q, vh2) = Q1 + Q2;

        // 更新顶点对
        for (auto heh : halfedge_mesh->voh_range(vh2)) {
            if (!halfedge_mesh->is_valid_handle(heh)) {
                continue;
            }
            auto vh = halfedge_mesh->to_vertex_handle(heh);
            // if (vh == vh1) {
            //     continue;
            // }
            if (halfedge_mesh->is_valid_handle(vh)) {
                VertexPair new_pair = create_vertex_pair(
                    halfedge_mesh, vprop_Q, vprop_version, vh2, vh);
                heap_vertex_pairs.push(new_pair);
            }
        }

        // 更新当前面数
        current_num_faces = 0;
        for (auto sf_it = halfedge_mesh->faces_sbegin();
             sf_it != halfedge_mesh->faces_end();
             ++sf_it) {
            current_num_faces++;
        }

    } while (current_num_faces > target_num_faces &&
             !heap_vertex_pairs.empty());
```

### 删除多余属性、清理无效顶点

```cpp
    // 删除Q矩阵属性
    halfedge_mesh->remove_property(vprop_Q);
    // 删除多余的顶点
    halfedge_mesh->garbage_collection();
```
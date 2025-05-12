# ARAP Deformation

## 创建矩阵

```cpp
	// 用三元组创建矩阵
	std::vector<Eigen::Triplet<double>> triple;

    // 将控制点设置为固定位置
    for (size_t i = 0; i < indices.size(); i++) {
        triple.push_back(Eigen::Triplet<double>(indices[i], indices[i], 1));
    }

	// 计算cot权重并创建矩阵
    for (auto eh : halfedge_mesh->edges()) {
        double cot_value = 0;
        int v0 = eh.v0().idx();
        int v1 = eh.v1().idx();
        for (auto efh : halfedge_mesh->ef_range(eh)) {
            int v2 = -1;
            for (auto fv : halfedge_mesh->fv_range(efh)) {
                if (fv.idx() != v0 && fv.idx() != v1) {
                    v2 = fv.idx();
                    break;
                }
            }
            assert(v2 != -1);
            auto edge0 =
                halfedge_mesh->point(halfedge_mesh->vertex_handle(v0)) -
                halfedge_mesh->point(halfedge_mesh->vertex_handle(v2));
            auto edge1 =
                halfedge_mesh->point(halfedge_mesh->vertex_handle(v1)) -
                halfedge_mesh->point(halfedge_mesh->vertex_handle(v2));
            double cos_value =
                edge0.dot(edge1) / (edge0.length() * edge1.length());
            double sin_value = sqrt(1 - cos_value * cos_value);
            cot_value += cos_value / sin_value;
        }
        cot_value *= 0.5;

        // 对控制点和非控制点区别处理
        bool v0_fixed =
            std::find(indices.begin(), indices.end(), v0) != indices.end();
        bool v1_fixed =
            std::find(indices.begin(), indices.end(), v1) != indices.end();
        if (v0_fixed && !v1_fixed) {
            triple.push_back(Eigen::Triplet<double>(v1, v1, cot_value));
            triple.push_back(Eigen::Triplet<double>(v1, v0, -cot_value));
        }
        else if (!v0_fixed && v1_fixed) {
            triple.push_back(Eigen::Triplet<double>(v0, v0, cot_value));
            triple.push_back(Eigen::Triplet<double>(v0, v1, -cot_value));
        }
        else if (!v0_fixed && !v1_fixed) {
            triple.push_back(Eigen::Triplet<double>(v0, v1, -cot_value));
            triple.push_back(Eigen::Triplet<double>(v1, v0, -cot_value));
            triple.push_back(Eigen::Triplet<double>(v0, v0, cot_value));
            triple.push_back(Eigen::Triplet<double>(v1, v1, cot_value));
        }
    }

    // 构建矩阵
    Eigen::SparseMatrix<double> A(n_vertices, n_vertices);
    A.setFromTriplets(triple.begin(), triple.end());

    // 预计算矩阵
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
```

## 存储原网格顶点位置，创建新网格顶点位置

```cpp
    // 将原网格顶点位置存储在P0，将变形后的网格位置存储在P
    Eigen::MatrixXd P0(n_vertices, 3), P(n_vertices, 3);
    for (auto vh : halfedge_mesh->vertices()) {
        int idx = vh.idx();
        auto& p = halfedge_mesh->point(vh);
        P0.row(idx) << p[0], p[1], p[2];
        P.row(idx) << p[0], p[1], p[2];
    }
    for (int i = 0; i < indices.size(); i++) {
        P.row(indices[i]) << new_positions[i][0], new_positions[i][1],
            new_positions[i][2];
    }
```

## 迭代求解

```cpp
    int max_iter = 10;
    int now_iter = 0;
    Eigen::VectorXd bx(n_vertices);
    Eigen::VectorXd by(n_vertices);
    Eigen::VectorXd bz(n_vertices);
    std::vector<Eigen::Matrix3d> Jacobi(n_faces);

    do
    {
        bx.setZero();
        by.setZero();
        bz.setZero();
        // Local Phase
        // ...
        // Global Phase
        // ...
    } while (now_iter < max_iter);
```

### Local Phase

固定顶点坐标，计算旋转矩阵`R`

```cpp
        // R为每个顶点的旋转矩阵
        std::vector<Eigen::Matrix3d> R(n_vertices, Eigen::Matrix3d::Identity());
        for (auto vh : halfedge_mesh->vertices()) {
            int i = vh.idx();
            if (std::find(indices.begin(), indices.end(), i) != indices.end()) {
                continue;  // Skip control points
            }
            // Ci = ∑_j w_ij * (P[i]-P[j])*(P0[i]-P0[j])^T
            Eigen::Matrix3d Ci = Eigen::Matrix3d::Zero();
            for (auto vih : halfedge_mesh->vv_range(vh)) {
                int j = vih.idx();
                double w = A.coeff(i, j) < 0 ? -A.coeff(i, j) : 0;
                Ci +=
                    w * (P.row(i).transpose() - P.row(j).transpose()) *
                    (P0.row(i).transpose() - P0.row(j).transpose()).transpose();
            }
            // SVD(Ci) -> Ri = U * V^T
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(
                Ci,
                Eigen::DecompositionOptions::ComputeThinU |
                    Eigen::DecompositionOptions::ComputeThinV);
            R[i] = svd.matrixU() * svd.matrixV().transpose();
        }
```

### Global Phase

固定旋转矩阵，计算新顶点坐标，并赋给`P`

```cpp
        for (auto vh : halfedge_mesh->vertices()) {
            int i = vh.idx();
            if (std::find(indices.begin(), indices.end(), i) != indices.end()) {
                bx(i) = P(i, 0);
                by(i) = P(i, 1);
                bz(i) = P(i, 2);
            }
            else {
                for (auto vih : halfedge_mesh->vv_range(vh)) {
                    int j = vih.idx();
                    double w = A.coeff(i, j) < 0 ? -A.coeff(i, j) : 0;
                    Eigen::Vector3d pij0 = (P0.row(i) - P0.row(j)).transpose();
                    Eigen::Vector3d rhs = 0.5 * w * (R[i] + R[j]) * pij0;
                    bx(i) += rhs(0);
                    by(i) += rhs(1);
                    bz(i) += rhs(2);
                }
            }
        }

        // Solve the linear equations
        Eigen::VectorXd ux = bx;
        ux = solver.solve(ux);
        Eigen::VectorXd uy = by;
        uy = solver.solve(uy);
        Eigen::VectorXd uz = bz;
        uz = solver.solve(uz);
        // Update vertex positions (P)
        for (int i = 0; i < n_vertices; i++) {
            if (std::find(indices.begin(), indices.end(), i) != indices.end()) {
                continue;  // Skip control points
            }
            P(i, 0) = ux(i);
            P(i, 1) = uy(i);
            P(i, 2) = uz(i);
        }
        now_iter++;
```
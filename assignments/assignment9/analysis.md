# Isotropic Remeshing

## 边分割

- 每条边分割后，会产生一个新的顶点和四条新边。

```cpp
void split_edges(std::shared_ptr<MyMesh> halfedge_mesh, float upper_bound)
{
    // TODO: split all edges at their midpoint that are longer then upper_bound
    // Create a set to store edges that need to be split
    std::unordered_set<MyMesh::EdgeHandle> edges;
    for (const auto& edge : halfedge_mesh->edges()) {
        if (halfedge_mesh->calc_edge_length(edge) > upper_bound) {
            edges.insert(edge);
        }
    }

    while (!edges.empty()) {
        MyMesh::EdgeHandle edge = *edges.begin();
        edges.erase(edges.begin());
        MyMesh::HalfedgeHandle heh = halfedge_mesh->halfedge_handle(edge, 0);
        if (!halfedge_mesh->is_valid_handle(heh)) {
            continue;  // Skip if halfedge handle is invalid
        }
        MyMesh::VertexHandle v1 = halfedge_mesh->to_vertex_handle(heh);
        MyMesh::VertexHandle v2 = halfedge_mesh->from_vertex_handle(heh);
        MyMesh::Point midpoint =
            (halfedge_mesh->point(v1) + halfedge_mesh->point(v2)) * 0.5f;
        MyMesh::VertexHandle new_vh = halfedge_mesh->add_vertex(midpoint);
        halfedge_mesh->split(edge, new_vh);
        // Four new edges are created, push them to the edge set
        for (const auto& new_edge : halfedge_mesh->ve_range(new_vh)) {
            if (halfedge_mesh->calc_edge_length(new_edge) > upper_bound) {
                edges.insert(new_edge);
            }
        }
    }
    halfedge_mesh->garbage_collection();
}
```

## 边合并

- 合并边后，需要在待处理边中删除与被合并边相交的边，并添加新产生的边。

```cpp
void collapse_edges(std::shared_ptr<MyMesh> halfedge_mesh, float lower_bound)
{
    // TODO: collapse all edges shorter than lower_bound into their midpoint
    // Create a set to store edges that need to be collapsed
    std::unordered_set<MyMesh::EdgeHandle> edges;
    for (const auto& edge : halfedge_mesh->edges()) {
        if (halfedge_mesh->calc_edge_length(edge) < lower_bound) {
            edges.insert(edge);
        }
    }

    while (!edges.empty()) {
        MyMesh::EdgeHandle edge = *edges.begin();
        edges.erase(edges.begin());
        MyMesh::HalfedgeHandle heh = halfedge_mesh->halfedge_handle(edge, 0);
        if (!halfedge_mesh->is_valid_handle(heh)) {
            continue;  // Skip if halfedge handle is invalid
        }
        if (!halfedge_mesh->is_collapse_ok(heh)) {
            continue;  // Skip if collapse is not valid
        }
        MyMesh::VertexHandle v1 = halfedge_mesh->to_vertex_handle(heh);
        MyMesh::VertexHandle v2 = halfedge_mesh->from_vertex_handle(heh);
        if (halfedge_mesh->is_boundary(v1) || halfedge_mesh->is_boundary(v2)) {
            continue;  // Skip if either vertex is a boundary vertex
        }
        for (const auto& adj_vh : halfedge_mesh->vv_range(v1)) {
            if (halfedge_mesh->is_boundary(adj_vh)) {
                continue;  // Skip if adjacent vertex is a boundary vertex
            }
        }
        for (const auto& adj_vh : halfedge_mesh->vv_range(v2)) {
            if (halfedge_mesh->is_boundary(adj_vh)) {
                continue;  // Skip if adjacent vertex is a boundary vertex
            }
        }

        // Remember old adjacent edges
        std::vector<MyMesh::EdgeHandle> adjacent_edges;
        for (const auto& adj_edge : halfedge_mesh->ve_range(v1)) {
            if (adj_edge != edge) {
                adjacent_edges.push_back(adj_edge);
            }
        }
        for (const auto& adj_edge : halfedge_mesh->ve_range(v2)) {
            if (adj_edge != edge) {
                adjacent_edges.push_back(adj_edge);
            }
        }

        // Collapse the edge into its midpoint
        MyMesh::Point midpoint =
            (halfedge_mesh->point(v1) + halfedge_mesh->point(v2)) * 0.5f;
        halfedge_mesh->set_point(v1, midpoint);
        halfedge_mesh->collapse(heh);

        // Erase old adjacent edges from the set
        for (const auto& adj_edge : adjacent_edges) {
            edges.erase(adj_edge);
        }
        // Add new adjacent edges to the set if they are shorter than
        // lower_bound
        for (const auto& new_edge : halfedge_mesh->ve_range(v1)) {
            if (halfedge_mesh->calc_edge_length(new_edge) < lower_bound) {
                edges.insert(new_edge);
            }
        }
    }
    halfedge_mesh->garbage_collection();
}
```

## 边翻转

- 翻转边后，原边的两个顶点分别度数减少1，新边的两个顶点分别度数增加1。

### 计算度数变化

```cpp
std::pair<int, int> compute_valence_excess(
    std::shared_ptr<MyMesh> halfedge_mesh,
    MyMesh::VertexHandle v1,
    MyMesh::VertexHandle v2,
    MyMesh::VertexHandle v3,
    MyMesh::VertexHandle v4)
{
    // Compute the valence excess for the vertices v1, v2, v3, and v4
    int valence_v1 = halfedge_mesh->valence(v1);
    int valence_v2 = halfedge_mesh->valence(v2);
    int valence_v3 = halfedge_mesh->valence(v3);
    int valence_v4 = halfedge_mesh->valence(v4);

    // Compute the excess for each vertex
    int excess_v1 = halfedge_mesh->is_boundary(v1) ? std::abs(valence_v1 - 4)
                                                   : std::abs(valence_v1 - 6);
    int excess_v2 = halfedge_mesh->is_boundary(v2) ? std::abs(valence_v2 - 4)
                                                   : std::abs(valence_v2 - 6);
    int excess_v3 = halfedge_mesh->is_boundary(v3) ? std::abs(valence_v3 - 4)
                                                   : std::abs(valence_v3 - 6);
    int excess_v4 = halfedge_mesh->is_boundary(v4) ? std::abs(valence_v4 - 4)
                                                   : std::abs(valence_v4 - 6);

    int excess_v1_flip = halfedge_mesh->is_boundary(v1)
                             ? std::abs(valence_v1 - 1 - 4)
                             : std::abs(valence_v1 - 1 - 6);
    int excess_v2_flip = halfedge_mesh->is_boundary(v2)
                             ? std::abs(valence_v2 - 1 - 4)
                             : std::abs(valence_v2 - 1 - 6);
    int excess_v3_flip = halfedge_mesh->is_boundary(v3)
                             ? std::abs(valence_v3 + 1 - 4)
                             : std::abs(valence_v3 + 1 - 6);
    int excess_v4_flip = halfedge_mesh->is_boundary(v4)
                             ? std::abs(valence_v4 + 1 - 4)
                             : std::abs(valence_v4 + 1 - 6);
    // Return the total excess for the vertices
    int total_excess = excess_v1 + excess_v2 + excess_v3 + excess_v4;
    int total_excess_flip =
        excess_v1_flip + excess_v2_flip + excess_v3_flip + excess_v4_flip;
    return std::make_pair(total_excess, total_excess_flip);
}
```

### 翻转边

- 使用`halfedge_mesh->is_flip_ok(edge)`方法检查翻转后是否会产生自交或其他问题。

```cpp
void flip_edges(std::shared_ptr<MyMesh> halfedge_mesh)
{
    // TODO: flip edges in order to minimize the deviation from valence 6
    // (or 4 on boundaries)
    for (const auto& edge : halfedge_mesh->edges()) {
        MyMesh::HalfedgeHandle heh = halfedge_mesh->halfedge_handle(edge, 0);
        if (halfedge_mesh->is_boundary(heh)) {
            continue;  // Skip boundary edges
        }
        if (!halfedge_mesh->is_flip_ok(edge)) {
            continue;  // Skip if flip is not valid
        }
        MyMesh::VertexHandle v1 = halfedge_mesh->to_vertex_handle(heh);
        MyMesh::VertexHandle v2 = halfedge_mesh->from_vertex_handle(heh);
        // Find the two vertices that will be connected after flipping
        MyMesh::HalfedgeHandle heh1 = halfedge_mesh->next_halfedge_handle(heh);
        MyMesh::HalfedgeHandle heh2 = halfedge_mesh->next_halfedge_handle(
            halfedge_mesh->opposite_halfedge_handle(heh));
        MyMesh::VertexHandle v3 = halfedge_mesh->to_vertex_handle(heh1);
        MyMesh::VertexHandle v4 = halfedge_mesh->to_vertex_handle(heh2);
        // Compute the total valence excess
        auto [total_excess, total_excess_flip] =
            compute_valence_excess(halfedge_mesh, v1, v2, v3, v4);
        // If flipping the edge reduces the total valence excess, flip it
        if (total_excess_flip < total_excess) {
            halfedge_mesh->flip(edge);
        }
    }
    halfedge_mesh->garbage_collection();
}
```

## 向重力加权质心（Gravity-Weighted Centroid）移动顶点

- 重力加权质心的计算公式为：
    $$
    g_i = \frac{1}{\sum_{p_j \in N(p_i)} A_{p_j}} \sum_{p_j \in N(p_i)} A_{p_j} p_j
    $$
- 向重力加权质心移动的公式为：
    $$
    p_i = p_i + \lambda \left( I - n_in_i^T \right) (g_i - p_i)
    $$

```cpp
MyMesh::Point calculate_normal(
    std::shared_ptr<MyMesh> halfedge_mesh,
    MyMesh::VertexHandle vertex)
{
    // Calculate the normal vector for a vertex based on its surrounding faces
    MyMesh::Normal normal(0.0f, 0.0f, 0.0f);
    for (const auto& face : halfedge_mesh->vf_range(vertex)) {
        normal += halfedge_mesh->calc_face_normal(face);
    }
    if (normal.norm() > 0.0f) {
        normal.normalize();
    }
    return normal;
}

void relocate_vertices(std::shared_ptr<MyMesh> halfedge_mesh, float lambda)
{
    // TODO: relocate vertices towards its gravity-weighted centroid
    std::vector<float> vertex_area(halfedge_mesh->n_vertices(), 0.0f);

    // Calculate area for each vertex (1/3 of sum of surrounding triangle areas)
    for (const auto& vertex : halfedge_mesh->vertices()) {
        float total_area = 0.0f;
        for (const auto& face : halfedge_mesh->vf_range(vertex)) {
            // Get the three vertices of the triangle
            auto fv_it = halfedge_mesh->fv_begin(face);
            MyMesh::Point p1 = halfedge_mesh->point(*fv_it++);
            MyMesh::Point p2 = halfedge_mesh->point(*fv_it++);
            MyMesh::Point p3 = halfedge_mesh->point(*fv_it);

            // Calculate triangle area using cross product
            MyMesh::Point v1 = p2 - p1;
            MyMesh::Point v2 = p3 - p1;
            float area = 0.5f * (v1 % v2).norm();
            total_area += area;
        }
        vertex_area[vertex.idx()] = total_area / 3.0f;
    }

    // Calculate gravity-weighted centroid for each vertex
    std::vector<MyMesh::Point> centroids(halfedge_mesh->n_vertices());
    for (const auto& vertex : halfedge_mesh->vertices()) {
        MyMesh::Point centroid(0.0f, 0.0f, 0.0f);
        float total_area = 0.0f;

        if (halfedge_mesh->is_boundary(vertex)) {
            // For boundary vertices, do not relocate
            centroids[vertex.idx()] = halfedge_mesh->point(vertex);
            continue;
        }

        for (const auto& vh : halfedge_mesh->vv_range(vertex)) {
            total_area += vertex_area[vh.idx()];
            centroid += halfedge_mesh->point(vh) * vertex_area[vh.idx()];
        }

        if (total_area > 0.0f) {
            centroid /= total_area;  // Normalize by total area
        }
        else {
            centroid =
                halfedge_mesh->point(vertex);  // Fallback to current position
        }
        centroids[vertex.idx()] = centroid;
    }

    // Tangentially move vertices towards their centroids
    for (const auto& vertex : halfedge_mesh->vertices()) {
        if (halfedge_mesh->is_boundary(vertex)) {
            continue;  // Skip boundary vertices
        }

        MyMesh::Point current_position = halfedge_mesh->point(vertex);  // p_i
        MyMesh::Point centroid = centroids[vertex.idx()];               // g_i
        // MyMesh::Point normal =
        //     halfedge_mesh->calc_vertex_normal(vertex);  // n_i
        MyMesh::Point normal = calculate_normal(halfedge_mesh, vertex);

        if (normal.norm() == 0.0f || std::isnan(normal[0]) ||
            std::isnan(normal[1]) || std::isnan(normal[2])) {
            // 添加调试信息
            std::cout << "Warning: Vertex " << vertex.idx()
                      << " has zero or NaN normal vector." << std::endl;

            std::cout << "Vertex " << vertex.idx()
                      << " position: " << current_position << std::endl;
            std::cout << "Vertex " << vertex.idx()
                      << " valence: " << halfedge_mesh->valence(vertex)
                      << std::endl;
            std::cout << "Raw normal: " << normal << " norm: " << normal.norm()
                      << std::endl;

            // 检查顶点周围的面
            int face_count = 0;
            for (const auto& face : halfedge_mesh->vf_range(vertex)) {
                face_count++;
                MyMesh::Point face_normal =
                    halfedge_mesh->calc_face_normal(face);
                std::cout << "  Face " << face.idx()
                          << " normal: " << face_normal << std::endl;
            }
            std::cout << "  Total faces around vertex: " << face_count
                      << std::endl;
            continue;  // Skip if normal is zero (e.g., isolated vertex)
        }

        // Normalize the normal vector
        normal.normalize();
        // std::cout << "normal: " << normal << std::endl;

        // Calculate (g_i - p_i)
        MyMesh::Point direction = centroid - current_position;

        // std::cout << "direction: " << direction << std::endl;

        // Calculate (I - n_i * n_i^T) * (g_i - p_i)
        // This is equivalent to: direction - (direction · n_i) * n_i
        float dot_product = direction | normal;  // dot product in OpenMesh
        // std::cout << "dot_product: " << dot_product << std::endl;
        MyMesh::Point tangential_component = direction - normal * dot_product;
        // std::cout << "tangential_component: " << tangential_component
        //           << std::endl;

        // Apply the formula: p_i = p_i + λ(I - n_i n_i^T)(g_i - p_i)
        MyMesh::Point new_position =
            current_position + lambda * tangential_component;

        // std::cout << "Relocating vertex " << vertex.idx() << " from "
        //           << current_position << " to " << new_position <<", the
        //           distance is "
        //           << (new_position - current_position).norm() << std::endl;

        halfedge_mesh->set_point(vertex, new_position);
    }
}
```


#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <boost/functional/hash.hpp>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <queue>
#include <unordered_map>

#include "Eigen/src/Core/Matrix.h"
#include "GCore/Components/MeshOperand.h"
#include "GCore/util_openmesh_bind.h"
#include "OpenMesh/Core/Mesh/Handles.hh"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;

struct VertexPair {
    MyMesh::VertexHandle v1;
    MyMesh::VertexHandle v2;
    int v1_version;
    int v2_version;
    Eigen::Vector3f optimal_point;
    float cost;

    bool operator>(const VertexPair& other) const
    {
        return cost > other.cost;
    }
};

Eigen::Vector4f compute_plane_equation(
    const std::shared_ptr<MyMesh>& mesh,
    MyMesh::FaceHandle fh)
{
    auto fv_it = mesh->fv_iter(fh);
    MyMesh::Point p0 = mesh->point(*fv_it);
    MyMesh::Point p1 = mesh->point(*++fv_it);
    MyMesh::Point p2 = mesh->point(*++fv_it);

    Eigen::Vector3f v0(p0[0], p0[1], p0[2]);
    Eigen::Vector3f v1(p1[0], p1[1], p1[2]);
    Eigen::Vector3f v2(p2[0], p2[1], p2[2]);

    Eigen::Vector3f e1 = v1 - v0;
    Eigen::Vector3f e2 = v2 - v0;
    Eigen::Vector3f normal = e1.cross(e2).normalized();

    float d = -normal.dot(v0);
    return Eigen::Vector4f(normal.x(), normal.y(), normal.z(), d);
}

float compute_cost(const Eigen::Matrix4f& Q, const Eigen::Vector3f& point)
{
    Eigen::Vector4f v(point.x(), point.y(), point.z(), 1.0f);
    return v.transpose() * Q * v;
}

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

void qem(
    std::shared_ptr<MyMesh> halfedge_mesh,
    float simplification_ratio,
    float distance_threshold)
{
    if (!halfedge_mesh || simplification_ratio <= 0 ||
        simplification_ratio >= 1)
        return;

    // 计算顶点对应的Q矩阵
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
            Eigen::Vector4f plane_eq =
                compute_plane_equation(halfedge_mesh, fh);
            Eigen::Matrix4f Q_i = plane_eq * plane_eq.transpose();
            Q += Q_i;
        }
        halfedge_mesh->property(vprop_Q, vh) = Q;
    }

    // 记录顶点版本号
    OpenMesh::VPropHandleT<int> vprop_version;
    halfedge_mesh->add_property(vprop_version);
    for (auto v_it = halfedge_mesh->vertices_begin();
         v_it != halfedge_mesh->vertices_end();
         ++v_it) {
        halfedge_mesh->property(vprop_version, *v_it) = 0;
    }

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
    // 删除Q矩阵属性
    halfedge_mesh->remove_property(vprop_Q);
    // 删除多余的顶点
    halfedge_mesh->garbage_collection();
}

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(qem)
{
    // Input-1: Original 3D mesh
    b.add_input<Geometry>("Input");
    // Input-2: Mesh simplification ratio, AKA the ratio of the number of
    // vertices in the simplified mesh to the number of vertices in the original
    // mesh
    b.add_input<float>("Simplification Ratio")
        .default_val(0.5f)
        .min(0.0f)
        .max(1.0f);
    // Input-3: Distance threshold for non-edge vertex pairs
    b.add_input<float>("Non-edge Distance Threshold")
        .default_val(0.01f)
        .min(0.0f)
        .max(1.0f);
    // Output-1: Simplified mesh
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(qem)
{
    // Get the input mesh
    auto input_mesh = params.get_input<Geometry>("Input");
    auto simplification_ratio = params.get_input<float>("Simplification Ratio");
    auto distance_threshold =
        params.get_input<float>("Non-edge Distance Threshold");

    // Avoid processing the node when there is no input
    if (!input_mesh.get_component<MeshComponent>()) {
        std::cerr << "QEM: No input mesh provided." << std::endl;
        return false;
    }

    /* ----------------------------- Preprocess
     *-------------------------------
     ** Create a halfedge structure (using OpenMesh) for the input mesh. The
     ** half-edge data structure is a widely used data structure in
     *geometric
     ** processing, offering convenient operations for traversing and
     *modifying
     ** mesh elements.
     */

    // Initialization
    auto halfedge_mesh = operand_to_openmesh(&input_mesh);

    halfedge_mesh->request_vertex_status();
    halfedge_mesh->request_edge_status();
    halfedge_mesh->request_face_status();
    halfedge_mesh->request_halfedge_status();

    // QEM simplification
    qem(halfedge_mesh, simplification_ratio, distance_threshold);

    // Convert the simplified mesh back to the operand
    auto geometry = openmesh_to_operand(halfedge_mesh.get());

    auto mesh = geometry->get_component<MeshComponent>();

    // Set the output of the nodes
    params.set_output("Output", std::move(*geometry));

    return true;
}

NODE_DECLARATION_UI(qem);

NODE_DEF_CLOSE_SCOPE

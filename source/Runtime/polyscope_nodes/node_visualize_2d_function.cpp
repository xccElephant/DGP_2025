#include <igl/triangle/triangulate.h>
#include <pxr/base/vt/types.h>

#include <Eigen/Core>
#include <functional>

#include "GCore/Components/MeshOperand.h"
#include "GCore/GOP.h"
#include "GCore/geom_payload.hpp"
#include "nodes/core/def/node_def.hpp"
#include "polyscope/surface_mesh.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/vt/array.h"

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(visualize_2d_function)
{
    // 输入一个XY平面上的二维多边形
    b.add_input<Geometry>("geometry");
    // 输入一个二维函数，返回一个标量
    b.add_input<std::function<float(float, float)>>("function");
    // 三角形最大面积的倒数
    b.add_input<int>("fineness").min(2).max(5).default_val(5);
    b.add_output<Geometry>("geometry");
}

NODE_EXECUTION_FUNCTION(visualize_2d_function)
{
    // 获取输入的二维顶点
    auto geometry = params.get_input<Geometry>("geometry");
    auto mesh = geometry.get_component<MeshComponent>();

    if (!mesh) {
        std::cerr << "Visualize 2D Function Node: Failed to get MeshComponent "
                     "from input geometry."
                  << std::endl;
        return false;
    }

    auto vertices = mesh->get_vertices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();
    auto face_vertex_indices = mesh->get_face_vertex_indices();

    // Ensure the input mesh is a 2D polygon
    if (vertices.size() < 3 || face_vertex_counts.size() != 1 ||
        face_vertex_counts[0] != vertices.size()) {
        std::cerr << "Visualize 2D Function Node: Input mesh must be a single "
                     "polygon with at least 3 vertices. "
                  << "Provided: " << vertices.size() << " vertices, "
                  << face_vertex_counts.size() << " faces. "
                  << "First face has "
                  << (face_vertex_counts.empty() ? 0 : face_vertex_counts[0])
                  << " vertices." << std::endl;

        return false;
    }

    // Ensure the polygon is on the XY plane
    for (const auto& vertex : vertices) {
        if (std::abs(vertex[2]) > 1e-5) {
            std::cerr
                << "Visualize 2D Function Node: Input mesh must be a 2D "
                   "polygon on the XY plane. Found vertex with Z-coordinate: "
                << vertex[2] << std::endl;
            return false;
        }
    }

    // 获取输入的二维函数
    auto function =
        params.get_input<std::function<float(float, float)>>("function");

    // 获取三角形最大面积的倒数
    auto fineness = params.get_input<int>("fineness");
    fineness = std::pow(10, fineness);

    // 将pxr::VtArray<pxr::GfVec3f>转换为Eigen矩阵
    Eigen::MatrixXf V(vertices.size(), 2);
    for (size_t i = 0; i < vertices.size(); ++i) {
        V(i, 0) = vertices[i][0];
        V(i, 1) = vertices[i][1];
    }

    // 创建边界顶点索引
    Eigen::MatrixXi E(vertices.size(), 2);
    for (size_t i = 0; i < vertices.size(); ++i) {
        E(i, 0) = i;
        E(i, 1) = (i + 1) % vertices.size();
    }

    // 使用libigl的triangulate函数生成三角形网格
    Eigen::MatrixXf V2;
    Eigen::MatrixXi F;
    std::string flags = "a" + std::to_string(1.0 / fineness) + "q";
    igl::triangle::triangulate(V, E, Eigen::MatrixXf(0, 2), flags, V2, F);

    // 将生成的三角形网格添加到polyscope中进行可视化
    // std::vector<std::array<size_t, 3>> faceVertexIndicesNested;
    // for (int i = 0; i < F.rows(); ++i) {
    //     faceVertexIndicesNested.push_back({ static_cast<size_t>(F(i, 0)),
    //                                         static_cast<size_t>(F(i, 1)),
    //                                         static_cast<size_t>(F(i, 2)) });
    // }

    // auto surface_mesh = polyscope::registerSurfaceMesh2D(
    //     sdf_path.GetName(), V2, faceVertexIndicesNested);

    // 将网格转换为Geometry格式
    pxr::VtArray<pxr::GfVec3f> vertices_2(V2.rows());
    pxr::VtArray<int> face_vertex_counts_2(F.rows());
    pxr::VtArray<int> face_vertex_indices_2(F.rows() * 3);
    for (int i = 0; i < V2.rows(); ++i) {
        vertices_2[i] = { V2(i, 0), V2(i, 1), 0.0f };
    }
    for (int i = 0; i < F.rows(); ++i) {
        face_vertex_counts_2[i] = 3;
        face_vertex_indices_2[i * 3] = F(i, 0);
        face_vertex_indices_2[i * 3 + 1] = F(i, 1);
        face_vertex_indices_2[i * 3 + 2] = F(i, 2);
    }

    Geometry geometry_2;
    auto mesh_2 = std::make_shared<MeshComponent>(&geometry_2);
    mesh_2->set_vertices(vertices_2);
    mesh_2->set_face_vertex_counts(face_vertex_counts_2);
    mesh_2->set_face_vertex_indices(face_vertex_indices_2);
    geometry_2.attach_component(mesh_2);

    // 添加顶点标量
    pxr::VtArray<float> vertex_scalar(V2.rows());
    for (int i = 0; i < V2.rows(); ++i) {
        vertex_scalar[i] = function(V2(i, 0), V2(i, 1));
    }
    // surface_mesh->addVertexScalarQuantity("function", vertex_scalar);

    mesh_2->add_vertex_scalar_quantity("function", vertex_scalar);

    params.set_output("geometry", geometry_2);

    return true;
}

NODE_DECLARATION_UI(visualize_2d_function);

NODE_DECLARATION_FUNCTION(random_2d_polygon)
{
    b.add_input<int>("Seed").min(0).max(10).default_val(0);
    b.add_input<int>("Vertices Count").min(3).max(10).default_val(4);
    b.add_output<Geometry>("Mesh");
}

NODE_EXECUTION_FUNCTION(random_2d_polygon)
{
    auto seed = params.get_input<int>("Seed");
    auto vertices_count = params.get_input<int>("Vertices Count");

    // 生成随机顶点位置，顶点个数为vertices_count，范围为[0, 1]
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    if (vertices_count < 3) {
        // 如果顶点数小于3，无法构成多边形，可以提前返回或报错
        // 根据节点定义，最小值为3，所以这里理论上不会发生
        std::cerr
            << "Random 2D Polygon Node: Vertices count must be at least 3."
            << std::endl;
        // 返回一个空几何体或者特定的错误状态
        params.set_output("Mesh", Geometry());
        return false;
    }

    // 1. 生成随机的二维点
    std::vector<pxr::GfVec2f> points_2d(vertices_count);
    for (int i = 0; i < vertices_count; ++i) {
        points_2d[i] = { dis(gen), dis(gen) };
    }

    // 2. 计算质心
    pxr::GfVec2f centroid(0.0f, 0.0f);
    for (const auto& p : points_2d) {
        centroid += p;
    }
    centroid /= static_cast<float>(vertices_count);

    // 3. 根据相对于质心的角度对点进行排序
    std::sort(
        points_2d.begin(),
        points_2d.end(),
        [&centroid](const pxr::GfVec2f& a, const pxr::GfVec2f& b) {
            float angle_a = atan2(a[1] - centroid[1], a[0] - centroid[0]);
            float angle_b = atan2(b[1] - centroid[1], b[0] - centroid[0]);
            return angle_a < angle_b;
        });

    // 4. 创建PXR顶点数组和面顶点索引数组
    pxr::VtArray<pxr::GfVec3f> vertices_pxr(vertices_count);
    pxr::VtArray<int> face_vertex_indices(vertices_count);
    for (int i = 0; i < vertices_count; ++i) {
        vertices_pxr[i] = { points_2d[i][0],
                            points_2d[i][1],
                            0.0f };  // Z坐标为0
        face_vertex_indices[i] = i;  // 顶点按排序后的顺序连接
    }

    Geometry geometry;
    auto mesh = std::make_shared<MeshComponent>(&geometry);
    mesh->set_vertices(vertices_pxr);
    // 多边形只有一个面，该面包含所有顶点
    mesh->set_face_vertex_counts({ vertices_count });
    mesh->set_face_vertex_indices(face_vertex_indices);
    geometry.attach_component(mesh);

    // 设置输出
    params.set_output("Mesh", geometry);

    return true;
}

NODE_DECLARATION_UI(random_2d_polygon);

NODE_DECLARATION_FUNCTION(function_decompose)
{
    b.add_input<std::function<std::vector<float>(float, float)>>("function");
    b.add_input<int>("index").min(0).max(10).default_val(0);
    b.add_output<std::function<float(float, float)>>("function");
}

NODE_EXECUTION_FUNCTION(function_decompose)
{
    auto function =
        params.get_input<std::function<std::vector<float>(float, float)>>(
            "function");
    auto index = params.get_input<int>("index");

    // 获取函数的分量
    auto function_component = function(0.0f, 0.0f);
    if (index < 0 || index >= function_component.size()) {
        std::cerr << "Index out of bounds for function decomposition."
                  << std::endl;
        return false;
    }

    // 创建新的函数，返回指定分量
    auto new_function = [function, index](float x, float y) {
        auto components = function(x, y);
        return components[index];
    };

    // 设置输出
    params.set_output(
        "function", std::function<float(float, float)>(new_function));

    return true;
}

NODE_DECLARATION_UI(function_decompose);

NODE_DEF_CLOSE_SCOPE

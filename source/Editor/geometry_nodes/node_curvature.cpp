#include <pxr/base/vt/array.h>

#include <vector>

#include "GCore/Components/MeshOperand.h"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

void compute_mean_curvature(
    const MyMesh& omesh,
    pxr::VtArray<float>& mean_curvature)
{
    // TODO: Implement the mean curvature computation
    //  You need to fill in `mean_curvature`
}

void compute_gaussian_curvature(
    const MyMesh& omesh,
    pxr::VtArray<float>& gaussian_curvature)
{
    // TODO: Implement the Gaussian curvature computation
    //  You need to fill in `gaussian_curvature`
}

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(mean_curvature)
{
    b.add_input<Geometry>("Mesh");
    b.add_output<pxr::VtArray<float>>("Mean Curvature");
}

NODE_EXECUTION_FUNCTION(mean_curvature)
{
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_indices = mesh->get_face_vertex_indices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();

    // Convert the mesh to OpenMesh
    MyMesh omesh;

    // Add vertices
    std::vector<OpenMesh::VertexHandle> vhandles;
    vhandles.reserve(vertices.size());

    for (auto vertex : vertices) {
        omesh.add_vertex(OpenMesh::Vec3f(vertex[0], vertex[1], vertex[2]));
    }

    // Add faces
    size_t start = 0;
    for (int face_vertex_count : face_vertex_counts) {
        std::vector<OpenMesh::VertexHandle> face;
        face.reserve(face_vertex_count);
        for (int j = 0; j < face_vertex_count; j++) {
            face.push_back(
                OpenMesh::VertexHandle(face_vertex_indices[start + j]));
        }
        omesh.add_face(face);
        start += face_vertex_count;
    }

    // Compute mean curvature
    pxr::VtArray<float> mean_curvature;
    mean_curvature.reserve(omesh.n_vertices());

    compute_mean_curvature(omesh, mean_curvature);

    params.set_output("Mean Curvature", mean_curvature);

    return true;
}

NODE_DECLARATION_UI(mean_curvature);

NODE_DECLARATION_FUNCTION(gaussian_curvature)
{
    b.add_input<Geometry>("Mesh");
    b.add_output<pxr::VtArray<float>>("Gaussian Curvature");
}

NODE_EXECUTION_FUNCTION(gaussian_curvature)
{
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_indices = mesh->get_face_vertex_indices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();

    // Convert the mesh to OpenMesh
    MyMesh omesh;

    // Add vertices
    std::vector<OpenMesh::VertexHandle> vhandles;
    vhandles.reserve(vertices.size());

    for (auto vertex : vertices) {
        omesh.add_vertex(OpenMesh::Vec3f(vertex[0], vertex[1], vertex[2]));
    }

    // Add faces
    size_t start = 0;
    for (int face_vertex_count : face_vertex_counts) {
        std::vector<OpenMesh::VertexHandle> face;
        face.reserve(face_vertex_count);
        for (int j = 0; j < face_vertex_count; j++) {
            face.push_back(
                OpenMesh::VertexHandle(face_vertex_indices[start + j]));
        }
        omesh.add_face(face);
        start += face_vertex_count;
    }

    // Compute Gaussian curvature
    pxr::VtArray<float> gaussian_curvature;
    gaussian_curvature.reserve(omesh.n_vertices());

    compute_gaussian_curvature(omesh, gaussian_curvature);

    params.set_output("Gaussian Curvature", gaussian_curvature);

    return true;
}

NODE_DECLARATION_UI(gaussian_curvature);

NODE_DEF_CLOSE_SCOPE

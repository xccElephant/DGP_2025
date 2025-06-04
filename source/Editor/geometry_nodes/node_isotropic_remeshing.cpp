#include <iostream>
#include <unordered_set>
#include <utility>

#include "GCore/Components/MeshOperand.h"
#include "GCore/util_openmesh_bind.h"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

void split_edges(std::shared_ptr<MyMesh> halfedge_mesh, float upper_bound)
{
    // TODO: split all edges at their midpoint that are longer then upper_bound
}

void collapse_edges(std::shared_ptr<MyMesh> halfedge_mesh, float lower_bound)
{
    // TODO: collapse all edges shorter than lower_bound into their midpoint
}

void flip_edges(std::shared_ptr<MyMesh> halfedge_mesh)
{
    // TODO: flip edges in order to minimize the deviation from valence 6
    // (or 4 on boundaries)
}

void relocate_vertices(std::shared_ptr<MyMesh> halfedge_mesh, float lambda)
{
    // TODO: relocate vertices towards its gravity-weighted centroid
}

void isotropic_remeshing(
    std::shared_ptr<MyMesh> halfedge_mesh,
    float target_edge_length,
    int num_iterations,
    float lambda)
{
    for (int i = 0; i < num_iterations; ++i) {
        split_edges(halfedge_mesh, target_edge_length * 4.0f / 3.0f);
        collapse_edges(halfedge_mesh, target_edge_length * 4.0f / 5.0f);
        flip_edges(halfedge_mesh);
        relocate_vertices(halfedge_mesh, lambda);
    }
}
NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(isotropic_remeshing)
{
    // The input-1 is a mesh
    b.add_input<Geometry>("Mesh");

    // The input-2 is the target edge length
    b.add_input<float>("Target Edge Length")
        .default_val(0.1f)
        .min(0.01f)
        .max(10.0f);

    // The input-3 is the number of iterations
    b.add_input<int>("Iterations").default_val(10).min(0).max(20);

    // The input-4 is the lambda value for vertex relocation
    b.add_input<float>("Lambda").default_val(1.0f).min(0.0f).max(1.0f);

    // The output is a remeshed version of the input mesh
    b.add_output<Geometry>("Remeshed Mesh");
}

NODE_EXECUTION_FUNCTION(isotropic_remeshing)
{
    // Get the input mesh
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();
    if (!mesh) {
        std::cerr << "Isotropic Remeshing Node: Failed to get MeshComponent "
                     "from input geometry."
                  << std::endl;
        return false;
    }
    auto halfedge_mesh = operand_to_openmesh_trimesh(&geometry);

    // Get the target edge length and number of iterations
    float target_edge_length = params.get_input<float>("Target Edge Length");
    int num_iterations = params.get_input<int>("Iterations");
    float lambda = params.get_input<float>("Lambda");
    if (target_edge_length <= 0.0f) {
        std::cerr << "Isotropic Remeshing Node: Target edge length must be "
                     "greater than zero."
                  << std::endl;
        return false;
    }

    if (num_iterations < 0) {
        std::cerr << "Isotropic Remeshing Node: Number of iterations must be "
                     "greater than zero."
                  << std::endl;
        return false;
    }

    halfedge_mesh->request_vertex_status();
    halfedge_mesh->request_edge_status();
    halfedge_mesh->request_face_status();
    halfedge_mesh->request_halfedge_status();
    halfedge_mesh->request_face_normals();
    halfedge_mesh->request_vertex_normals();
    halfedge_mesh->request_halfedge_normals();

    // Isotropic remeshing
    isotropic_remeshing(
        halfedge_mesh, target_edge_length, num_iterations, lambda);

    // Convert the remeshed halfedge mesh back to Geometry
    auto remeshed_geometry = openmesh_to_operand_trimesh(halfedge_mesh.get());
    // Set the output of the node
    params.set_output("Remeshed Mesh", std::move(*remeshed_geometry));

    return true;
}

NODE_DECLARATION_UI(isotropic_remeshing);
NODE_DEF_CLOSE_SCOPE

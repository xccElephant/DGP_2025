
#include <pxr/base/vt/array.h>

#include <vector>

#include "Eigen/Core"
#include "GCore/Components/CurveComponent.h"
#include "GCore/Components/MeshOperand.h"
#include "GCore/GOP.h"
#include "GCore/util_openmesh_bind.h"
#include "nodes/core/def/node_def.hpp"
#include "polyscope_widget/polyscope_renderer.h"

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(visualize_n_pv_field)
{
    // Input-1: Original 3D mesh
    b.add_input<Geometry>("Original Mesh");
    // Input-2: N-PolyVector field
    b.add_input<std::vector<std::vector<Eigen::Vector3d>>>(
        "N-PolyVector Field");
    // Output-1: Node curve
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(visualize_n_pv_field)
{
    // Get the input
    auto input_mesh = params.get_input<Geometry>("Original Mesh");
    auto n_pv_field =
        params.get_input<std::vector<std::vector<Eigen::Vector3d>>>(
            "N-PolyVector Field");

    // Avoid processing the node when there is no input
    if (!input_mesh.get_component<MeshComponent>()) {
        std::cerr << "N-PolyVector field visualizer: No input mesh provided."
                  << std::endl;
        return false;
    }

    auto mesh = input_mesh.get_component<MeshComponent>();

    // Check if the N-PolyVector field is valid
    if (n_pv_field.empty() ||
        n_pv_field.size() != mesh->get_face_vertex_counts().size()) {
        std::cerr
            << "N-PolyVector field visualizer: Invalid N-PolyVector field."
            << std::endl;
        return false;
    }

    pxr::VtArray<pxr::GfVec3f> vertices;
    pxr::VtArray<int> vertex_counts;

    auto halfedge_mesh = operand_to_openmesh(&input_mesh);

    for (auto it = halfedge_mesh->faces_sbegin();
         it != halfedge_mesh->faces_end();
         it++) {
        auto fh = *it;
        auto centroid = halfedge_mesh->calc_face_centroid(fh);
        auto frame_fields = n_pv_field[fh.idx()];
        for (const auto& field : frame_fields) {
            vertices.push_back({ static_cast<float>(centroid[0]),
                                 static_cast<float>(centroid[1]),
                                 static_cast<float>(centroid[2]) });
            vertices.push_back({ static_cast<float>(centroid[0] + field[0]),
                                 static_cast<float>(centroid[1] + field[1]),
                                 static_cast<float>(centroid[2] + field[2]) });
            vertex_counts.push_back(2);
        }
    }

    // Create a new Geometry to hold the output
    Geometry output_geometry;
    auto curve_component = std::make_shared<CurveComponent>(&output_geometry);
    output_geometry.attach_component(curve_component);
    curve_component->set_vertices(vertices);
    curve_component->set_vert_count(vertex_counts);

    // Set the output
    params.set_output("Output", output_geometry);

    return true;
}

NODE_DECLARATION_UI(visualize_n_pv_field);

NODE_DEF_CLOSE_SCOPE

#include "GCore/Components/MeshOperand.h"
#include "GCore/util_openmesh_bind.h"
#include "cross_field_reference.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(cross_field)
{
    // Input-1: Original 3D mesh
    b.add_input<Geometry>("Input");
    // Output-1: Cross field
    b.add_output<std::vector<std::vector<Eigen::Vector3d>>>("Cross Field");
}

NODE_EXECUTION_FUNCTION(cross_field)
{  // Get the input mesh
    auto input_mesh = params.get_input<Geometry>("Input");

    // Avoid processing the node when there is no input
    if (!input_mesh.get_component<MeshComponent>()) {
        std::cerr << "Cross field: No input mesh provided." << std::endl;
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
    auto openmesh = std::make_shared<EigenPolyMesh>();
    auto topology = input_mesh.get_component<MeshComponent>();

    for (const auto& vv : topology->get_vertices()) {
        Eigen::Vector3d v(vv[0], vv[1], vv[2]);
        openmesh->add_vertex(v);
    }

    auto faceVertexIndices = topology->get_face_vertex_indices();
    auto faceVertexCounts = topology->get_face_vertex_counts();

    int vertexIndex = 0;
    for (int i = 0; i < faceVertexCounts.size(); i++) {
        // Create a vector of vertex handles for the face
        std::vector<EigenPolyMesh::VertexHandle> face_vhandles;
        for (int j = 0; j < faceVertexCounts[i]; j++) {
            int index = faceVertexIndices[vertexIndex];
            // Get the vertex handle from the index
            EigenPolyMesh::VertexHandle vh = openmesh->vertex_handle(index);
            // Add it to the vector
            face_vhandles.push_back(vh);
            vertexIndex++;
        }
        // Add the face to the mesh
        openmesh->add_face(face_vhandles);
    }

    // Create the cross field
    CrossField cross_field(*openmesh);
    cross_field.generateBoundaryConstrain();
    cross_field.generateCrossField();
    auto cross_fields = cross_field.getCrossFields();

    // Set the output
    params.set_output("Cross Field", cross_fields);

    return true;
}

NODE_DECLARATION_UI(cross_field);

NODE_DEF_CLOSE_SCOPE

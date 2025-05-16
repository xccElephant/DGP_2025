#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
#include <iostream>
#include <memory>

#include "GCore/Components/MeshOperand.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;

void qem(
    std::shared_ptr<MyMesh> halfedge_mesh,
    float simplification_ratio,
    float distance_threshold)
{
    // TODO: Implement QEM mesh simplification algorithm.

    /*
     * QEM (Quadric Error Metrics) Mesh Simplification Algorithm
     *
     * 1. Compute the Q matrices for all the initial vertices.
     *    - For each vertex, accumulate the quadric error matrices derived from
     *      the planes of its adjacent faces.
     *    - The quadric for a face is constructed from its plane equation.
     *
     * 2. Select all valid pairs.
     *    - For each edge (v1, v2), consider the pair as valid for contraction.
     *    - Optional: Consider non-edge vertex pairs based on distance
     *      threshold.
     *
     * 3. Compute the optimal contraction target v_bar for each valid pair
     *    (v1, v2).
     *    - The optimal position minimizes the quadric error
     *      v_bar^T (Q1 + Q2) v_bar.
     *    - The cost of contracting the pair is the error at this optimal
     *      position.
     *
     * 4. Place all the pairs in a heap keyed on cost with the minimum cost pair
     *    at the top.
     *    - Use a priority queue to efficiently extract the pair with the lowest
     *      cost.
     *
     * 5. Iteratively remove the pair (v1, v2) of least cost from the heap,
     *    contract this pair, and update the costs of all valid pairs involving
     *    v1.
     *    - After contraction, update the affected quadrics and heap entries.
     *
     * 6. Iterate until the number of faces in the mesh is less than
     *    simplification ratio * number of faces in the original mesh.
     *
     * The initial Q matrices are computed by summing the outer products of the
     * plane equations of all faces adjacent to each vertex.
     */

    // Placeholder. You need to implement the QEM algorithm here.
    return;
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

    /* ----------------------------- Preprocess -------------------------------
     ** Create a halfedge structure (using OpenMesh) for the input mesh. The
     ** half-edge data structure is a widely used data structure in geometric
     ** processing, offering convenient operations for traversing and modifying
     ** mesh elements.
     */

    // Initialization
    auto halfedge_mesh = operand_to_openmesh(&input_mesh);

    // QEM simplification
    qem(halfedge_mesh, simplification_ratio, distance_threshold);

    // Convert the simplified mesh back to the operand
    auto geometry = openmesh_to_operand(halfedge_mesh.get());

    // Set the output of the nodes
    params.set_output("Output", std::move(*geometry));

    return true;
}

NODE_DECLARATION_UI(qem);

NODE_DEF_CLOSE_SCOPE

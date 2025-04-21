#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "Eigen/src/Core/Matrix.h"
#include "GCore/Components/MeshOperand.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;

void arap_deformation(
    std::shared_ptr<MyMesh> halfedge_mesh,
    std::vector<size_t> indices,
    std::vector<std::array<float, 3>> new_positions)
{
    // TODO: Implement ARAP Deformation Algorithm.

    /* ------------- ARAP Deformation Implementation -----------
     ** Implement ARAP mesh deformation to preserve local rigidity.
     **
     ** Steps:
     ** 1. Initial Setup:
     **    - Build cotangent‐weighted Laplacian L of the input mesh.
     **    - Apply Dirichlet constraints: for each control index i, enforce
     **      L(i,i)=1.
     **
     ** 2. Pre‐Factorization:
     **    - Factorize L (e.g. with Eigen::SparseLU) once for fast solves.
     **
     ** 3. Initialization:
     **    - Record original positions P0 (n×3).
     **    - Initialize deformed positions P = P0.
     **    - Enforce control points: for each t, P(indices[t]) =
     **      new_positions[t].
     **
     ** 4. Iteration (Local–Global):
     **    Local Phase:
     **      For each vertex i ∉ control:
     **        • Compute covariance Ci = Σ_{j∈N(i)} w_ij (P_i−P_j)(P0_i−P0_j)^T.
     **        • SVD(Ci) → U,Σ,Vᵀ; set rotation R_i = U·Vᵀ.
     **
     **    Global Phase:
     **      • Assemble RHS vectors bx,by,bz (size n):
     **          – If i ∈ control, b*(i)=P(i,*).
     **          – Else b*(i)=½ Σ_{j∈N(i)} w_ij·(R_i+R_j)·(P0_i−P0_j).
     **      • Solve L·ux=bx, L·uy=by, L·uz=bz.
     **      • Update free P_i = (ux_i,uy_i,uz_i); re‐enforce control points.
     **
     **    Repeat until max_iter or convergence.
     **
     ** 5. Finalization:
     **    - Write P back into halfedge_mesh→point(vh) for all vertices.
     **
     */
}

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(arap_deformation)
{
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");

    // Input-2: Indices of the control vertices
    b.add_input<std::vector<size_t>>("Indices");

    // Input-3: New positions for the control vertices
    b.add_input<std::vector<std::array<float, 3>>>("New Positions");

    // Output-1: Deformed mesh
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(arap_deformation)
{
    // Get the input from params
    auto input = params.get_input<Geometry>("Input");
    auto indices = params.get_input<std::vector<size_t>>("Indices");
    auto new_positions =
        params.get_input<std::vector<std::array<float, 3>>>("New Positions");

    if (indices.empty() || new_positions.empty()) {
        std::cerr << "ARAP Deformation: Please set control points."
                  << std::endl;
        return false;
    }

    if (indices.size() != new_positions.size()) {
        std::cerr << "ARAP Deformation: The size of indices and new positions "
                     "should be the same."
                  << std::endl;
        return false;
    }

    /* ----------------------------- Preprocess -------------------------------
     ** Create a halfedge structure (using OpenMesh) for the input mesh. The
     ** half-edge data structure is a widely used data structure in geometric
     ** processing, offering convenient operations for traversing and modifying
     ** mesh elements.
     */

    // Initialization
    auto halfedge_mesh = operand_to_openmesh(&input);

    // ARAP deformation
    arap_deformation(halfedge_mesh, indices, new_positions);

    auto geometry = openmesh_to_operand(halfedge_mesh.get());

    // Set the output of the nodes
    params.set_output("Output", std::move(*geometry));
    return true;
}

NODE_DECLARATION_UI(arap_deformation);
NODE_DEF_CLOSE_SCOPE

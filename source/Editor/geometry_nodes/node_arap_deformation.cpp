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
using Eigen::Matrix3d;
using Eigen::SparseLU;
using Eigen::SparseMatrix;
using Eigen::Vector3d;
using Eigen::VectorXd;

// Helper function to calculate cotangent weights
double calculate_cotangent(const MyMesh& mesh,MyMesh::HalfedgeHandle he)
{
    if(mesh.is_boundary(he) || mesh.is_boundary(mesh.opposite_halfedge_handle(he))) {
        return 0.0;
    }

    auto p0 = mesh.point(mesh.to_vertex_handle(he));
    auto p1 = mesh.point(mesh.from_vertex_handle(he));
    auto p2 = mesh.point(mesh.to_vertex_handle(mesh.next_halfedge_handle(he)));
    auto p3 = mesh.point(
        mesh.to_vertex_handle(mesh.next_halfedge_handle(mesh.opposite_halfedge_handle(he))));

    auto v1 = p0 - p2;
    auto v2 = p1 - p2;
    auto v3 = p0 - p3;
    auto v4 = p1 - p3;

    double cot_alpha = v1.dot(v2) / v1.cross(v2).norm();
    double cot_beta = v3.dot(v4) / v3.cross(v4).norm();

    return (cot_alpha + cot_beta) / 2.0;
}

void arap_deformation(
    std::shared_ptr<MyMesh> halfedge_mesh,
    std::vector<size_t> indices,
    std::vector<std::array<float,3>> new_positions)
{
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

    const int n_vertices = halfedge_mesh->n_vertices();
    const int max_iters = 5; // Maximum number of iterations

    // Convert control point indices to a set for faster lookup
    std::vector<bool> is_control(n_vertices,false);
    for(size_t idx : indices) {
        if(idx < n_vertices) {
            is_control[idx] = true;
        }
    }

    // Step 1: Initial Setup - Build cotangent‐weighted Laplacian L
    SparseMatrix<double> L(n_vertices,n_vertices);
    std::vector<Eigen::Triplet<double>> L_triplets;
    L_triplets.reserve(n_vertices * 7); // Pre-allocate approximate space

    for(auto vh : halfedge_mesh->vertices()) {
        int i = vh.idx();
        if(is_control[i]) {
            L_triplets.emplace_back(i,i,1.0);
        } else {
            double weight_sum = 0.0;
            for(auto voh_it = halfedge_mesh->cvoh_iter(vh); voh_it.is_valid(); ++voh_it) {
                int j = halfedge_mesh->to_vertex_handle(*voh_it).idx();
                double wij = calculate_cotangent(*halfedge_mesh,*voh_it);
                // Clamp weights to avoid numerical issues with degenerate triangles
                wij = std::max(0.0,wij);
                L_triplets.emplace_back(i,j,-wij);
                weight_sum += wij;
            }
            // Handle potential isolated vertices or issues with weights
            if(weight_sum == 0.0) weight_sum = 1.0;
            L_triplets.emplace_back(i,i,weight_sum);
        }
    }
    L.setFromTriplets(L_triplets.begin(),L_triplets.end());

    // Step 2: Pre‐Factorization
    SparseLU<SparseMatrix<double>> solver;
    solver.compute(L);
    if(solver.info() != Eigen::Success) {
        std::cerr << "ARAP Deformation: SparseLU factorization failed." << std::endl;
        // Potentially return or handle error appropriately
        return;
    }

    // Step 3: Initialization
    std::vector<Vector3d> P0(n_vertices); // Original positions
    std::vector<Vector3d> P(n_vertices);  // Deformed positions
    for(auto vh : halfedge_mesh->vertices()) {
        int i = vh.idx();
        const auto& pt = halfedge_mesh->point(vh);
        P0[i] = Vector3d(pt[0],pt[1],pt[2]);
        P[i] = P0[i]; // Initialize P = P0
    }

    // Enforce control point constraints initially
    for(size_t k = 0; k < indices.size(); ++k) {
        size_t i = indices[k];
        if(i < n_vertices) {
            P[i] = Vector3d(new_positions[k][0],new_positions[k][1],new_positions[k][2]);
        }
    }


    // Step 4: Iteration (Local–Global)
    std::vector<Matrix3d> R(n_vertices); // Rotation matrices

    for(int iter = 0; iter < max_iters; ++iter) {
        // --- Local Phase ---
        for(auto vh : halfedge_mesh->vertices()) {
            int i = vh.idx();
            if(is_control[i]) {
                R[i] = Matrix3d::Identity(); // Control points don't need rotation estimation
                continue;
            }

            Matrix3d Ci = Matrix3d::Zero();
            for(auto voh_it = halfedge_mesh->cvoh_iter(vh); voh_it.is_valid(); ++voh_it) {
                int j = halfedge_mesh->to_vertex_handle(*voh_it).idx();
                double wij = calculate_cotangent(*halfedge_mesh,*voh_it);
                wij = std::max(0.0,wij); // Clamp weight

                Vector3d Pi_Pj = P[i] - P[j];
                Vector3d P0i_P0j = P0[i] - P0[j];

                Ci += wij * (Pi_Pj * P0i_P0j.transpose());
            }

            // SVD on covariance matrix Ci = U S V^T
            Eigen::JacobiSVD<Matrix3d> svd(Ci,Eigen::ComputeFullU | Eigen::ComputeFullV);
            Matrix3d U = svd.matrixU();
            Matrix3d V = svd.matrixV();
            Matrix3d Ri = V * U.transpose(); // Note: Sorkine paper uses R = VU^T

            // Ensure determinant is positive (handle reflection)
            if(Ri.determinant() < 0) {
                // V.col(2) *= -1; // Adjust last column of V
                // Ri = V * U.transpose();
                U.col(2) *= -1; // Adjust last column of U is more stable
                Ri = V * U.transpose();
            }
            R[i] = Ri;
        }

        // --- Global Phase ---
        VectorXd bx = VectorXd::Zero(n_vertices);
        VectorXd by = VectorXd::Zero(n_vertices);
        VectorXd bz = VectorXd::Zero(n_vertices);

        for(auto vh : halfedge_mesh->vertices()) {
            int i = vh.idx();
            if(is_control[i]) {
                // RHS for control points is their fixed position
                bx(i) = P[i].x();
                by(i) = P[i].y();
                bz(i) = P[i].z();
            } else {
                Vector3d bi = Vector3d::Zero();
                for(auto voh_it = halfedge_mesh->cvoh_iter(vh); voh_it.is_valid(); ++voh_it) {
                    int j = halfedge_mesh->to_vertex_handle(*voh_it).idx();
                    double wij = calculate_cotangent(*halfedge_mesh,*voh_it);
                    wij = std::max(0.0,wij); // Clamp weight

                    Vector3d P0i_P0j = P0[i] - P0[j];
                    Matrix3d Rij_avg = R[i] + R[j];

                    bi += (wij / 2.0) * Rij_avg * P0i_P0j;
                }
                bx(i) = bi.x();
                by(i) = bi.y();
                bz(i) = bi.z();
            }
        }

        // Solve linear systems: L * P_new = b
        VectorXd Px_new = solver.solve(bx);
        VectorXd Py_new = solver.solve(by);
        VectorXd Pz_new = solver.solve(bz);

        if(solver.info() != Eigen::Success) {
            std::cerr << "ARAP Deformation: Solver failed in iteration " << iter << std::endl;
            // Potentially break or handle error
            break;
        }

        // Update positions for non-control vertices
        for(int i = 0; i < n_vertices; ++i) {
            // Re-enforce control points strictly after solve
            if(!is_control[i]) {
                P[i] = Vector3d(Px_new(i),Py_new(i),Pz_new(i));
            }
        }
        // Re-enforce control points strictly after solve
        for(size_t k = 0; k < indices.size(); ++k) {
            size_t i = indices[k];
            if(i < n_vertices) {
                P[i] = Vector3d(new_positions[k][0],new_positions[k][1],new_positions[k][2]);
            }
        }
    }

    // Step 5: Finalization - Write deformed positions P back to the mesh
    for(auto vh : halfedge_mesh->vertices()) {
        int i = vh.idx();
        halfedge_mesh->set_point(vh,MyMesh::Point(P[i].x(),P[i].y(),P[i].z()));
    }
}

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(arap_deformation)
{
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");

    // Input-2: Indices of the control vertices
    b.add_input<std::vector<size_t>>("Indices");

    // Input-3: New positions for the control vertices
    b.add_input<std::vector<std::array<float,3>>>("New Positions");

    // Output-1: Deformed mesh
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(arap_deformation)
{
    // Get the input from params
    auto input = params.get_input<Geometry>("Input");
    auto indices = params.get_input<std::vector<size_t>>("Indices");
    auto new_positions =
        params.get_input<std::vector<std::array<float,3>>>("New Positions");

    if(indices.empty() || new_positions.empty()) {
        std::cerr << "ARAP Deformation: Please set control points."
            << std::endl;
        return false;
    }

    if(indices.size() != new_positions.size()) {
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
    arap_deformation(halfedge_mesh,indices,new_positions);

    auto geometry = openmesh_to_operand(halfedge_mesh.get());

    // Set the output of the nodes
    params.set_output("Output",std::move(*geometry));
    return true;
}

NODE_DECLARATION_UI(arap_deformation);
NODE_DEF_CLOSE_SCOPE

#include <igl/arap.h>
#include <igl/cotmatrix.h>
#include <igl/min_quad_with_fixed.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "GCore/Components/MeshOperand.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "igl/ARAPEnergyType.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;

void arap_deformation(
    std::shared_ptr<MyMesh> halfedge_mesh,
    std::vector<size_t> indices,
    std::vector<std::array<float, 3>> new_positions)
{
    int n_faces = halfedge_mesh->n_faces();
    int n_vertices = halfedge_mesh->n_vertices();

    // Eigen::SparseMatrix<double> cotangents(n_vertices, n_vertices);
    std::vector<Eigen::Triplet<double>> triple;

    // Set the fixed vertex (control point)
    for (size_t i = 0; i < indices.size(); i++) {
        triple.push_back(Eigen::Triplet<double>(indices[i], indices[i], 1));
    }

    for (auto eh : halfedge_mesh->edges()) {
        double cot_value = 0;
        int v0 = eh.v0().idx();
        int v1 = eh.v1().idx();
        for (auto efh : halfedge_mesh->ef_range(eh)) {
            int v2 = -1;
            for (auto fv : halfedge_mesh->fv_range(efh)) {
                if (fv.idx() != v0 && fv.idx() != v1) {
                    v2 = fv.idx();
                    break;
                }
            }
            assert(v2 != -1);
            auto edge0 =
                halfedge_mesh->point(halfedge_mesh->vertex_handle(v0)) -
                halfedge_mesh->point(halfedge_mesh->vertex_handle(v2));
            auto edge1 =
                halfedge_mesh->point(halfedge_mesh->vertex_handle(v1)) -
                halfedge_mesh->point(halfedge_mesh->vertex_handle(v2));
            double cos_value =
                edge0.dot(edge1) / (edge0.length() * edge1.length());
            double sin_value = sqrt(1 - cos_value * cos_value);
            cot_value += cos_value / sin_value;
        }
        cot_value *= 0.5;

        // Check if the vertex is a control point
        bool v0_fixed =
            std::find(indices.begin(), indices.end(), v0) != indices.end();
        bool v1_fixed =
            std::find(indices.begin(), indices.end(), v1) != indices.end();
        if (v0_fixed && !v1_fixed) {
            triple.push_back(Eigen::Triplet<double>(v1, v1, cot_value));
            triple.push_back(Eigen::Triplet<double>(v1, v0, -cot_value));
        }
        else if (!v0_fixed && v1_fixed) {
            triple.push_back(Eigen::Triplet<double>(v0, v0, cot_value));
            triple.push_back(Eigen::Triplet<double>(v0, v1, -cot_value));
        }
        else if (!v0_fixed && !v1_fixed) {
            triple.push_back(Eigen::Triplet<double>(v0, v1, -cot_value));
            triple.push_back(Eigen::Triplet<double>(v1, v0, -cot_value));
            triple.push_back(Eigen::Triplet<double>(v0, v0, cot_value));
            triple.push_back(Eigen::Triplet<double>(v1, v1, cot_value));
        }
    }

    // Construct the matrix of the function, which can be precomputed before
    // iteration
    Eigen::SparseMatrix<double> A(n_vertices, n_vertices);
    A.setFromTriplets(triple.begin(), triple.end());

    // Precompute the matrix
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);

    // Set original vertex positions (P0) and deformed vertex positions (P)
    Eigen::MatrixXd P0(n_vertices, 3), P(n_vertices, 3);
    for (auto vh : halfedge_mesh->vertices()) {
        int idx = vh.idx();
        auto& p = halfedge_mesh->point(vh);
        P0.row(idx) << p[0], p[1], p[2];
        P.row(idx) << p[0], p[1], p[2];
    }
    for (int i = 0; i < indices.size(); i++) {
        P.row(indices[i]) << new_positions[i][0], new_positions[i][1],
            new_positions[i][2];
    }

    int max_iter = 10;
    int now_iter = 0;
    // double err_pre = -1e9;
    // double err = 1e9;
    Eigen::VectorXd bx(n_vertices);
    Eigen::VectorXd by(n_vertices);
    Eigen::VectorXd bz(n_vertices);
    std::vector<Eigen::Matrix3d> Jacobi(n_faces);

    do {
        bx.setZero();
        by.setZero();
        bz.setZero();
        // Local phase
        // Rotation matrix for each vertex (R)
        std::vector<Eigen::Matrix3d> R(n_vertices, Eigen::Matrix3d::Identity());
        for (auto vh : halfedge_mesh->vertices()) {
            int i = vh.idx();
            if (std::find(indices.begin(), indices.end(), i) != indices.end()) {
                continue;  // Skip control points
            }
            // Ci = ∑_j w_ij * (P[i]-P[j])*(P0[i]-P0[j])^T
            Eigen::Matrix3d Ci = Eigen::Matrix3d::Zero();
            for (auto vih : halfedge_mesh->vv_range(vh)) {
                int j = vih.idx();
                double w = A.coeff(i, j) < 0 ? -A.coeff(i, j) : 0;
                Ci +=
                    w * (P.row(i).transpose() - P.row(j).transpose()) *
                    (P0.row(i).transpose() - P0.row(j).transpose()).transpose();
            }
            // SVD(Ci) -> Ri = U * V^T
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(
                Ci,
                Eigen::DecompositionOptions::ComputeThinU |
                    Eigen::DecompositionOptions::ComputeThinV);
            R[i] = svd.matrixU() * svd.matrixV().transpose();
        }

        // Global phase
        for (auto vh : halfedge_mesh->vertices()) {
            int i = vh.idx();
            if (std::find(indices.begin(), indices.end(), i) != indices.end()) {
                bx(i) = P(i, 0);
                by(i) = P(i, 1);
                bz(i) = P(i, 2);
            }
            else {
                for (auto vih : halfedge_mesh->vv_range(vh)) {
                    int j = vih.idx();
                    double w = A.coeff(i, j) < 0 ? -A.coeff(i, j) : 0;
                    Eigen::Vector3d pij0 = (P0.row(i) - P0.row(j)).transpose();
                    Eigen::Vector3d rhs = 0.5 * w * (R[i] + R[j]) * pij0;
                    bx(i) += rhs(0);
                    by(i) += rhs(1);
                    bz(i) += rhs(2);
                }
            }
        }

        // Solve the linear equations
        Eigen::VectorXd ux = bx;
        ux = solver.solve(ux);
        Eigen::VectorXd uy = by;
        uy = solver.solve(uy);
        Eigen::VectorXd uz = bz;
        uz = solver.solve(uz);
        // Update vertex positions (P)
        for (int i = 0; i < n_vertices; i++) {
            if (std::find(indices.begin(), indices.end(), i) != indices.end()) {
                continue;  // Skip control points
            }
            P(i, 0) = ux(i);
            P(i, 1) = uy(i);
            P(i, 2) = uz(i);
        }
        now_iter++;
    } while (now_iter < max_iter);

    // Update the mesh with the new vertex positions
    for (auto vh : halfedge_mesh->vertices()) {
        int idx = vh.idx();
        halfedge_mesh->point(vh)[0] = static_cast<float>(P(idx, 0));
        halfedge_mesh->point(vh)[1] = static_cast<float>(P(idx, 1));
        halfedge_mesh->point(vh)[2] = static_cast<float>(P(idx, 2));
    }
}

void arap_deformation_libigl(
    std::shared_ptr<MyMesh> halfedge_mesh,
    std::vector<size_t> indices,
    std::vector<std::array<float, 3>> new_positions)
{
    // Step 1: Convert mesh to Eigen matrices
    int n_vertices = halfedge_mesh->n_vertices();
    int n_faces = halfedge_mesh->n_faces();

    // Initial vertex positions
    Eigen::MatrixXd V(n_vertices, 3);
    // Face indices
    Eigen::MatrixXi F(n_faces, 3);

    // Fill vertex matrix
    for (auto vh : halfedge_mesh->vertices()) {
        int idx = vh.idx();
        auto p = halfedge_mesh->point(vh);
        V(idx, 0) = p[0];
        V(idx, 1) = p[1];
        V(idx, 2) = p[2];
    }

    // Fill face matrix
    int face_idx = 0;
    for (auto fh : halfedge_mesh->faces()) {
        int vertex_idx = 0;
        for (auto fv_it = halfedge_mesh->cfv_iter(fh); fv_it.is_valid();
             ++fv_it) {
            F(face_idx, vertex_idx) = fv_it->idx();
            vertex_idx++;
        }
        face_idx++;
    }

    // Step 2: Set up fixed vertices and target positions
    Eigen::VectorXi b(indices.size());
    Eigen::MatrixXd bc(indices.size(), 3);

    for (size_t i = 0; i < indices.size(); i++) {
        b(i) = static_cast<int>(indices[i]);
        bc(i, 0) = new_positions[i][0];
        bc(i, 1) = new_positions[i][1];
        bc(i, 2) = new_positions[i][2];
    }

    // Step 3: Set up ARAP solver
    igl::ARAPData arap_data;
    arap_data.energy = igl::ARAP_ENERGY_TYPE_SPOKES;
    // arap_data.max_iter = 300;  // Maximum iterations for convergence

    // Initialize ARAP (precomputation step)
    if (!igl::arap_precomputation(V, F, 3, b, arap_data)) {
        std::cerr << "ARAP precomputation failed." << std::endl;
        return;
    }

    // Step 4: Solve for deformation
    Eigen::MatrixXd U =
        V;  // Initialize deformed vertices with original positions
    if (!igl::arap_solve(bc, arap_data, U)) {
        std::cerr << "ARAP solve failed." << std::endl;
        return;
    }

    // Step 5: Update mesh with deformed positions
    for (auto vh : halfedge_mesh->vertices()) {
        int idx = vh.idx();
        halfedge_mesh->point(vh)[0] = static_cast<float>(U(idx, 0));
        halfedge_mesh->point(vh)[1] = static_cast<float>(U(idx, 1));
        halfedge_mesh->point(vh)[2] = static_cast<float>(U(idx, 2));
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

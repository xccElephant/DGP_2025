#include <Eigen/Sparse>
#include <cmath>

#include "GCore/Components/MeshOperand.h"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "geom_node_base.h"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

void tutte_embedding(MyMesh& omesh)
{
    // TODO: Implement Tutte Embedding Algorithm.
    //
    // In this task, you are required to **modify** the original mesh to a
    // 'minimal surface' mesh with the boundary of the input mesh as its
    // boundary.
    //
    // Specifically, the positions of the boundary vertices of the input mesh
    // should be fixed. By solving a global Laplace equation on the mesh,
    // recalculate the coordinates of the vertices inside the mesh to achieve
    // the minimal surface configuration

    /*
     ** Algorithm Pseudocode for Minimal Surface Calculation
     ** ------------------------------------------------------------------------
     ** 1. Initialize mesh with input boundary conditions.
     **    - For each boundary vertex, fix its position.
     **    - For internal vertices, initialize with initial guess if necessary.
     **
     ** 2. Construct Laplacian matrix for the mesh.
     **    - Compute weights for each edge based on the chosen weighting scheme
     **      (e.g., uniform weights for simplicity).
     **    - Assemble the global Laplacian matrix.
     **
     ** 3. Solve the Laplace equation for interior vertices.
     **    - Apply Dirichlet boundary conditions for boundary vertices.
     **    - Solve the linear system (Laplacian * X = 0) to find new positions
     **      for internal vertices.
     **
     ** 4. Update mesh geometry with new vertex positions.
     **    - Ensure the mesh respects the minimal surface configuration.
     **
     ** Note: This pseudocode outlines the general steps for calculating a
     ** minimal surface mesh given fixed boundary conditions using the Laplace
     ** equation. The specific implementation details may vary based on the mesh
     ** representation and numerical methods used.
     **
     */

    // Step 1: Identify boundary and interior vertices
    std::vector<bool> is_boundary(omesh.n_vertices(), false);
    std::vector<int> interior_indices;
    std::vector<int> boundary_indices;

    for (auto v_it = omesh.vertices_begin(); v_it != omesh.vertices_end(); ++v_it) {
        if (omesh.is_boundary(*v_it)) {
            is_boundary[v_it->idx()] = true;
            boundary_indices.push_back(v_it->idx());
        } else {
            interior_indices.push_back(v_it->idx());
        }
    }

    int num_interior = interior_indices.size();
    int num_boundary = boundary_indices.size();
    int num_vertices = omesh.n_vertices();

    // Create a mapping from vertex indices to matrix indices for interior vertices
    std::map<int, int> index_map;
    for (int i = 0; i < num_interior; ++i) {
        index_map[interior_indices[i]] = i;
    }

    // Step 2: Construct the Laplacian matrix (using uniform weights)
    typedef Eigen::Triplet<double> T;
    std::vector<T> triplets;
    
    // Step 3: Set up the right-hand sides for x, y, and z coordinates
    Eigen::VectorXd b_x = Eigen::VectorXd::Zero(num_interior);
    Eigen::VectorXd b_y = Eigen::VectorXd::Zero(num_interior);
    Eigen::VectorXd b_z = Eigen::VectorXd::Zero(num_interior);

    // For each interior vertex, construct the Laplacian row
    for (int i = 0; i < num_interior; ++i) {
        int v_idx = interior_indices[i];
        auto v_handle = omesh.vertex_handle(v_idx);
        
        // Count the neighbors of the vertex
        int valence = 0;
        for (auto vv_it = omesh.vv_iter(v_handle); vv_it.is_valid(); ++vv_it) {
            valence++;
        }
        
        // Set diagonal element to valence (number of neighbors)
        triplets.push_back(T(i, i, valence));
        
        // Process each neighbor
        for (auto vv_it = omesh.vv_iter(v_handle); vv_it.is_valid(); ++vv_it) {
            int neighbor_idx = vv_it->idx();
            
            // Set -1 for each neighbor
            if (!is_boundary[neighbor_idx]) {
                // Interior neighbor contributes to the Laplacian matrix
                int j = index_map[neighbor_idx];
                triplets.push_back(T(i, j, -1.0));
            } else {
                // Boundary neighbor contributes to the right-hand side
                OpenMesh::Vec3f p = omesh.point(*vv_it);
                b_x(i) += p[0];
                b_y(i) += p[1];
                b_z(i) += p[2];
            }
        }
    }

    // Create sparse matrix
    Eigen::SparseMatrix<double> L(num_interior, num_interior);
    L.setFromTriplets(triplets.begin(), triplets.end());

    // Solve the system for each coordinate
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(L);
    
    if (solver.info() != Eigen::Success) {
        // Handle error
        std::cerr << "Decomposition failed!" << std::endl;
        return;
    }

    Eigen::VectorXd x = solver.solve(b_x);
    Eigen::VectorXd y = solver.solve(b_y);
    Eigen::VectorXd z = solver.solve(b_z);

    if (solver.info() != Eigen::Success) {
        // Handle error
        std::cerr << "Solving failed!" << std::endl;
        return;
    }

    // Step 4: Update the mesh with new vertex positions
    for (int i = 0; i < num_interior; ++i) {
        int v_idx = interior_indices[i];
        auto v_handle = omesh.vertex_handle(v_idx);
        omesh.set_point(v_handle, OpenMesh::Vec3f(x(i), y(i), z(i)));
    }

    return;
}

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(tutte)
{
    // Function content omitted
    b.add_input<Geometry>("Input");

    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(tutte)
{
    // Function content omitted

    // Get the input from params
    auto input = params.get_input<Geometry>("Input");

    // Avoid processing the node when there is no input
    if (!input.get_component<MeshComponent>()) {
        std::cerr << "Tutte Parameterization: Need Geometry Input."
                  << std::endl;
        return false;
    }

    auto mesh = input.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_indices = mesh->get_face_vertex_indices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();

    // Convert the mesh to OpenMesh
    MyMesh omesh;

    for (int i = 0; i < vertices.size(); i++) {
        omesh.add_vertex(
            OpenMesh::Vec3f(vertices[i][0], vertices[i][1], vertices[i][2]));
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

    omesh.request_vertex_normals();
    omesh.request_face_normals();
    omesh.update_normals();

    // Perform Tutte Embedding
    tutte_embedding(omesh);

    // Convert back to Geometry
    pxr::VtArray<pxr::GfVec3f> tutte_vertices;
    for (const auto& v : omesh.vertices()) {
        const auto& p = omesh.point(v);
        tutte_vertices.push_back(pxr::GfVec3f(p[0], p[1], p[2]));
    }
    pxr::VtArray<int> tutte_faceVertexIndices;
    pxr::VtArray<int> tutte_faceVertexCounts;
    for (const auto& f : omesh.faces()) {
        size_t count = 0;
        for (const auto& vf : f.vertices()) {
            tutte_faceVertexIndices.push_back(vf.idx());
            count += 1;
        }
        tutte_faceVertexCounts.push_back(count);
    }

    Geometry tutte_geometry;
    auto tutte_mesh = std::make_shared<MeshComponent>(&tutte_geometry);

    tutte_mesh->set_vertices(tutte_vertices);
    tutte_mesh->set_face_vertex_indices(tutte_faceVertexIndices);
    tutte_mesh->set_face_vertex_counts(tutte_faceVertexCounts);
    tutte_geometry.attach_component(tutte_mesh);
    // Set the output of the nodes
    params.set_output("Output", tutte_geometry);
    return true;
}

NODE_DECLARATION_UI(tutte);
NODE_DEF_CLOSE_SCOPE

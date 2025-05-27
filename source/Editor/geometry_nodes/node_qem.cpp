#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <algorithm> // Required for std::min/max, std::sort

#include "GCore/Components/MeshOperand.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;

// Structure to store per-vertex Q matrix
// Each vertex will have associated data like its Q matrix.
struct VertexData {
    Eigen::Matrix4d q_matrix;
    VertexData() : q_matrix(Eigen::Matrix4d::Zero()) {}
};

// Structure for an edge collapse operation candidate
// It stores the vertices of the edge, the optimal target position for collapse,
// the cost of this collapse, and a unique ID for priority queue management.
struct EdgeCollapse {
    MyMesh::VertexHandle v0_handle; // One vertex of the edge (typically the one to be kept/modified)
    MyMesh::VertexHandle v1_handle; // The other vertex (typically the one to be removed)
    Eigen::Vector4d optimal_pos_homogeneous; // Optimal position (x, y, z, 1) for the new vertex
    double cost; // The quadric error cost of this collapse
    int unique_id; // A unique ID to handle updates in the priority queue

    // Comparator for the priority queue (min-heap based on cost)
    // Lower cost means higher priority. Tie-breaking with unique_id.
    bool operator>(const EdgeCollapse& other) const {
        if (cost != other.cost) {
            return cost > other.cost;
        }
        return unique_id > other.unique_id; // Ensures stable sort for equal costs
    }
};


// Helper function to calculate the Quadric Error Matrix (Q) for a single face.
// The Q matrix for a plane ax + by + cz + d = 0 is pp^T, where p = [a, b, c, d]^T.
Eigen::Matrix4d calculate_face_q_matrix(const MyMesh* mesh, MyMesh::FaceHandle fh) {
    std::vector<MyMesh::Point> face_vertex_points;
    // Collect vertex coordinates of the face
    for (MyMesh::ConstFaceVertexIter cfv_it = mesh->cfv_iter(fh); cfv_it.is_valid(); ++cfv_it) {
        face_vertex_points.push_back(mesh->point(*cfv_it));
    }

    // A face must have at least 3 vertices to define a plane
    if (face_vertex_points.size() < 3) {
        return Eigen::Matrix4d::Zero(); // Return zero matrix for degenerate faces
    }

    // Convert OpenMesh points to Eigen vectors for calculations
    Eigen::Vector3d p0(face_vertex_points[0][0], face_vertex_points[0][1], face_vertex_points[0][2]);
    Eigen::Vector3d p1(face_vertex_points[1][0], face_vertex_points[1][1], face_vertex_points[1][2]);
    Eigen::Vector3d p2(face_vertex_points[2][0], face_vertex_points[2][1], face_vertex_points[2][2]);

    // Calculate the normal of the plane defined by the first three vertices
    Eigen::Vector3d normal = (p1 - p0).cross(p2 - p0);
    if (normal.norm() < 1e-9) { // Check for collinear vertices / degenerate triangle
        return Eigen::Matrix4d::Zero();
    }
    normal.normalize(); // Ensure unit normal

    // Plane equation: ax + by + cz + d = 0
    // a, b, c are components of the normal vector
    // d = -normal.dot(p0) (or any point on the plane)
    double a = normal.x();
    double b = normal.y();
    double c = normal.z();
    double d_coeff = -normal.dot(p0);

    Eigen::Vector4d plane_params(a, b, c, d_coeff);
    // K_p = p * p^T
    return plane_params * plane_params.transpose();
}

// Helper function to calculate the optimal contraction target position and its associated cost.
// Given Q_sum = Q1 + Q2 for a pair (v1, v2), find v_bar that minimizes v_bar^T * Q_sum * v_bar.
// Returns a pair: {optimal_pos_homogeneous, cost}
std::pair<Eigen::Vector4d, double> calculate_optimal_contraction(
    const Eigen::Matrix4d& Q_sum,        // Sum of Q matrices for the pair (Q1 + Q2)
    const MyMesh::Point& p_v0_geom,      // Original position of vertex v0
    const MyMesh::Point& p_v1_geom) {    // Original position of vertex v1

    Eigen::Matrix4d Q_prime = Q_sum;
    // To find the optimal position (x,y,z), we solve Q_prime * [x,y,z,1]^T = [0,0,0,k]^T (error derivative)
    // This is equivalent to solving the top 3 rows for x,y,z:
    // [q11 q12 q13] [x]   [-q14]
    // [q21 q22 q23] [y] = [-q24]
    // [q31 q32 q33] [z]   [-q34]
    // We modify the last row of Q_prime to [0,0,0,1] to make it invertible for finding [x,y,z,1]^T if needed,
    // but the primary method involves solving the 3x3 system.
    
    Eigen::Matrix3d A = Q_sum.topLeftCorner<3, 3>();
    Eigen::Vector3d b_vector = -Q_sum.block<3, 1>(0, 3); // Corresponds to -[q14, q24, q34]^T

    Eigen::Vector4d optimal_pos_h; // Homogeneous coordinates (x, y, z, 1)
    double min_cost;

    if (std::abs(A.determinant()) > 1e-9) { // If A (the 3x3 part of Q_sum) is invertible
        Eigen::Vector3d optimal_xyz = A.inverse() * b_vector;
        optimal_pos_h << optimal_xyz, 1.0;
    } else {
        // Fallback strategy if A is singular (cannot invert)
        // Test v0, v1, and their midpoint (v0+v1)/2. Choose the one with the minimum error.
        Eigen::Vector4d v0_h, v1_h, mid_h;
        v0_h << p_v0_geom[0], p_v0_geom[1], p_v0_geom[2], 1.0;
        v1_h << p_v1_geom[0], p_v1_geom[1], p_v1_geom[2], 1.0;
        mid_h << (p_v0_geom[0] + p_v1_geom[0]) / 2.0,
                 (p_v0_geom[1] + p_v1_geom[1]) / 2.0,
                 (p_v0_geom[2] + p_v1_geom[2]) / 2.0,
                 1.0;

        double cost_v0 = v0_h.transpose() * Q_sum * v0_h;
        double cost_v1 = v1_h.transpose() * Q_sum * v1_h;
        double cost_mid = mid_h.transpose() * Q_sum * mid_h;

        if (cost_v0 < cost_v1 && cost_v0 < cost_mid) {
            optimal_pos_h = v0_h;
        } else if (cost_v1 < cost_mid) {
            optimal_pos_h = v1_h;
        } else {
            optimal_pos_h = mid_h;
        }
    }
    min_cost = optimal_pos_h.transpose() * Q_sum * optimal_pos_h;
    return {optimal_pos_h, min_cost};
}


void qem(
    std::shared_ptr<MyMesh> halfedge_mesh,
    float simplification_ratio,
    float distance_threshold) // Note: distance_threshold for non-edge pairs is not implemented here.
{
    // Basic check for a valid mesh
    if (!halfedge_mesh || halfedge_mesh->n_vertices() == 0) {
        std::cerr << "QEM: Input mesh is null or empty." << std::endl;
        return;
    }

    // Request status bits if not already available (important for deletions)
    halfedge_mesh->request_vertex_status();
    halfedge_mesh->request_edge_status();
    halfedge_mesh->request_face_status();

    // Add a custom property to vertices to store their Q matrices.
    // This avoids using a separate map and integrates well with OpenMesh.
    OpenMesh::VPropHandleT<VertexData> v_data_prop;
    if (!halfedge_mesh->get_property_handle(v_data_prop, "v_data_qem")) { // Use a unique name
         halfedge_mesh->add_property(v_data_prop, "v_data_qem");
    }
    
    // 1. Compute the Q matrices for all initial vertices.
    // For each vertex, its Q matrix is the sum of Q matrices of its adjacent faces.
    for (MyMesh::VertexIter v_it = halfedge_mesh->vertices_begin(); v_it != halfedge_mesh->vertices_end(); ++v_it) {
        MyMesh::VertexHandle vh = *v_it;
        halfedge_mesh->property(v_data_prop, vh).q_matrix.setZero(); // Initialize Q matrix to zero
        // Iterate over faces adjacent to the current vertex
        for (MyMesh::ConstVertexFaceIter cvf_it = halfedge_mesh->cvf_iter(vh); cvf_it.is_valid(); ++cvf_it) {
            halfedge_mesh->property(v_data_prop, vh).q_matrix += calculate_face_q_matrix(halfedge_mesh.get(), *cvf_it);
        }
    }

    // Priority queue to store potential edge collapses, ordered by cost (min-heap).
    std::priority_queue<EdgeCollapse, std::vector<EdgeCollapse>, std::greater<EdgeCollapse>> pq;
    
    // Map to track unique IDs of edge collapses currently in the PQ.
    // Key: pair of vertex indices (min_idx, max_idx) to uniquely identify an edge.
    // Value: unique_id of the EdgeCollapse operation in the PQ.
    // This helps in "invalidating" old entries when an edge's cost is recomputed.
    std::map<std::pair<int, int>, int> edge_to_active_pq_id;
    std::set<int> invalidated_pq_ids; // Stores unique_ids of operations that are no longer valid.
    int next_available_pq_id = 0;

    // 2. Select all valid pairs (edges in this case) and compute their contraction cost.
    // 3. Place all pairs in the heap (priority queue).
    for (MyMesh::EdgeIter e_it = halfedge_mesh->edges_begin(); e_it != halfedge_mesh->edges_end(); ++e_it) {
        MyMesh::EdgeHandle eh = *e_it;
        if (halfedge_mesh->status(eh).deleted()) continue; // Skip deleted edges

        MyMesh::HalfedgeHandle heh0 = halfedge_mesh->halfedge_handle(eh, 0);
        // MyMesh::HalfedgeHandle heh1 = halfedge_mesh->halfedge_handle(eh, 1); // opposite direction

        MyMesh::VertexHandle vh0 = halfedge_mesh->from_vertex_handle(heh0);
        MyMesh::VertexHandle vh1 = halfedge_mesh->to_vertex_handle(heh0);

        // Ensure consistent ordering for map keys and storage (v0_handle < v1_handle by index)
        MyMesh::VertexHandle v_first = (vh0.idx() < vh1.idx()) ? vh0 : vh1;
        MyMesh::VertexHandle v_second = (vh0.idx() < vh1.idx()) ? vh1 : vh0;

        Eigen::Matrix4d Q_sum = halfedge_mesh->property(v_data_prop, v_first).q_matrix +
                                halfedge_mesh->property(v_data_prop, v_second).q_matrix;
        
        auto contraction_result = calculate_optimal_contraction(Q_sum, halfedge_mesh->point(v_first), halfedge_mesh->point(v_second));
        
        EdgeCollapse current_collapse;
        current_collapse.v0_handle = v_first;  // Vertex to be kept/modified
        current_collapse.v1_handle = v_second; // Vertex to be removed
        current_collapse.optimal_pos_homogeneous = contraction_result.first;
        current_collapse.cost = contraction_result.second;
        current_collapse.unique_id = next_available_pq_id++;
        
        pq.push(current_collapse);
        edge_to_active_pq_id[{v_first.idx(), v_second.idx()}] = current_collapse.unique_id;
    }

    // Calculate initial and target number of faces
    // As per README, count faces by iterating, as n_faces() might not be up-to-date without garbage_collection.
    int initial_num_faces = 0;
    for (auto f_it = halfedge_mesh->faces_sbegin(); f_it != halfedge_mesh->faces_end(); ++f_it) {
        if (!halfedge_mesh->status(*f_it).deleted()) {
            initial_num_faces++;
        }
    }
    int target_num_faces = static_cast<int>(initial_num_faces * simplification_ratio);
    // Ensure at least one simplification if ratio is 1.0 but mesh has faces, or if ratio is very close to 1.
    if (target_num_faces >= initial_num_faces && initial_num_faces > 0) target_num_faces = initial_num_faces -1; 
    if (target_num_faces < 0) target_num_faces = 0; // Cannot have negative faces

    int current_num_faces = initial_num_faces;
    int iterations_count = 0;
    const int max_allowed_iterations = initial_num_faces * 3; // Safety break to prevent infinite loops

    // 5. Iteratively remove the pair (v_kept, v_removed) of least cost, contract, and update.
    while (current_num_faces > target_num_faces && !pq.empty() && iterations_count < max_allowed_iterations) {
        EdgeCollapse best_collapse_candidate = pq.top();
        pq.pop();
        iterations_count++;

        // Validity checks for the candidate from PQ:
        // - Has its unique_id been marked as invalidated?
        // - Are both its vertices still present (not deleted)?
        // - Is this the most up-to-date collapse operation for this edge pair?
        if (invalidated_pq_ids.count(best_collapse_candidate.unique_id) ||
            halfedge_mesh->status(best_collapse_candidate.v0_handle).deleted() ||
            halfedge_mesh->status(best_collapse_candidate.v1_handle).deleted() ||
            best_collapse_candidate.v0_handle == best_collapse_candidate.v1_handle) { // v0_handle can become == v1_handle if an endpoint was collapsed into the other
            continue; // Skip this outdated or invalid collapse operation
        }
        
        // Check if this is the active PQ entry for this edge
        std::pair<int, int> edge_key = {best_collapse_candidate.v0_handle.idx(), best_collapse_candidate.v1_handle.idx()};
        if (edge_to_active_pq_id.find(edge_key) == edge_to_active_pq_id.end() ||
            edge_to_active_pq_id[edge_key] != best_collapse_candidate.unique_id) {
            continue; // Not the most current collapse for this edge, skip.
        }


        MyMesh::VertexHandle v_kept = best_collapse_candidate.v0_handle;
        MyMesh::VertexHandle v_removed = best_collapse_candidate.v1_handle;

        // Find a halfedge representing the edge (v_removed, v_kept) to perform the collapse.
        // OpenMesh's `collapse` operation removes `from_vertex_handle(heh)` and merges it into `to_vertex_handle(heh)`.
        // So, we need a halfedge from `v_removed` to `v_kept`.
        MyMesh::HalfedgeHandle heh_for_collapse;
        bool heh_found = false;
        for (MyMesh::VOHIter voh_it = halfedge_mesh->voh_iter(v_removed); voh_it.is_valid(); ++voh_it) {
            if (halfedge_mesh->to_vertex_handle(*voh_it) == v_kept) {
                heh_for_collapse = *voh_it;
                heh_found = true;
                break;
            }
        }
        if (!heh_found) { // Should ideally always find it if they form an edge and are valid.
                          // This might happen if topology changed unexpectedly.
            invalidated_pq_ids.insert(best_collapse_candidate.unique_id); // Mark as invalid just in case
            edge_to_active_pq_id.erase(edge_key);
            continue;
        }
        
        // Check if OpenMesh allows this collapse (e.g., manifoldness check)
        if (!halfedge_mesh->is_collapse_ok(heh_for_collapse)) {
            invalidated_pq_ids.insert(best_collapse_candidate.unique_id); // This specific collapse is not OK
            edge_to_active_pq_id.erase(edge_key); // Remove from active map
            continue; 
        }

        // Perform the collapse:
        // 1. Set the position of the kept vertex to the optimal position.
        MyMesh::Point new_pos_geom(best_collapse_candidate.optimal_pos_homogeneous[0],
                                   best_collapse_candidate.optimal_pos_homogeneous[1],
                                   best_collapse_candidate.optimal_pos_homogeneous[2]);
        halfedge_mesh->set_point(v_kept, new_pos_geom);

        // 2. Perform the topological collapse. v_removed is merged into v_kept.
        //    v_removed and incident edges/faces are deleted or updated.
        halfedge_mesh->collapse(heh_for_collapse);

        // 3. Update the Q matrix for the (now modified) kept vertex.
        //    Q_new = Q_original_v_kept + Q_original_v_removed
        //    The Q matrices for v_kept and v_removed were stored in best_collapse_candidate implicitly via their sum.
        //    We need the individual original Q matrices to sum them.
        //    Or, more directly, the sum Q_sum used to calculate the cost IS the new Q matrix.
         halfedge_mesh->property(v_data_prop, v_kept).q_matrix = halfedge_mesh->property(v_data_prop, best_collapse_candidate.v0_handle).q_matrix +
                                                               halfedge_mesh->property(v_data_prop, best_collapse_candidate.v1_handle).q_matrix;
                                                               // This uses the property values from before this collapse, which is correct.

        // 4. Update costs for edges incident to the new v_kept.
        //    - Invalidate old entries in PQ related to v_kept and v_removed.
        //    - Add new entries for edges now connected to v_kept.
        
        // Collect neighbors of the modified v_kept to recompute edge costs
        std::set<MyMesh::VertexHandle> neighbors_of_v_kept;
        for (MyMesh::ConstVertexVertexIter cvv_it = halfedge_mesh->cvv_iter(v_kept); cvv_it.is_valid(); ++cvv_it) {
            if (!halfedge_mesh->status(*cvv_it).deleted()) {
                neighbors_of_v_kept.insert(*cvv_it);
            }
        }
        
        // Invalidate all old edge pairs involving v_kept or v_removed from the active map and PQ invalidation set.
        // This is because their topology or one of their endpoints (v_removed) has changed.
        // Iterate over a temporary list of neighbors of the *original* v_kept and v_removed if needed,
        // but since v_removed is gone, we focus on new neighbors of v_kept.
        // The previous active map entry for (v_kept, v_removed) is already handled.
        // We need to update/invalidate pairs (v_kept, u) for all neighbors u.

        for (MyMesh::VertexHandle neighbor_vh : neighbors_of_v_kept) {
            MyMesh::VertexHandle u = neighbor_vh;
            // Ensure consistent order for edge key
            MyMesh::VertexHandle v_first_new = (v_kept.idx() < u.idx()) ? v_kept : u;
            MyMesh::VertexHandle v_second_new = (v_kept.idx() < u.idx()) ? u : v_kept;
            std::pair<int, int> new_edge_key = {v_first_new.idx(), v_second_new.idx()};

            // If this edge (v_first_new, v_second_new) had an old entry in PQ, invalidate it.
            if (edge_to_active_pq_id.count(new_edge_key)) {
                invalidated_pq_ids.insert(edge_to_active_pq_id[new_edge_key]);
                // No need to erase from map yet, will be overwritten by new entry or skipped if not added
            }
            
            // Recompute cost for the edge (v_kept, u)
            Eigen::Matrix4d Q_sum_new = halfedge_mesh->property(v_data_prop, v_first_new).q_matrix +
                                        halfedge_mesh->property(v_data_prop, v_second_new).q_matrix;
            auto new_contraction_result = calculate_optimal_contraction(Q_sum_new, halfedge_mesh->point(v_first_new), halfedge_mesh->point(v_second_new));

            EdgeCollapse updated_collapse;
            updated_collapse.v0_handle = v_first_new;
            updated_collapse.v1_handle = v_second_new;
            updated_collapse.optimal_pos_homogeneous = new_contraction_result.first;
            updated_collapse.cost = new_contraction_result.second;
            updated_collapse.unique_id = next_available_pq_id++;

            pq.push(updated_collapse);
            edge_to_active_pq_id[new_edge_key] = updated_collapse.unique_id; // Track the new active ID for this edge
        }
        
        // Update face count
        current_num_faces = 0;
        for (auto f_it = halfedge_mesh->faces_sbegin(); f_it != halfedge_mesh->faces_end(); ++f_it) {
             if (!halfedge_mesh->status(*f_it).deleted()) {
                current_num_faces++;
             }
        }
    }

    // Perform garbage collection to remove deleted elements and update mesh indices.
    // The README advises caution, but it's generally needed for a clean final mesh.
    halfedge_mesh->garbage_collection();

    // Output some statistics
    int final_face_count = 0;
     for (auto f_it = halfedge_mesh->faces_sbegin(); f_it != halfedge_mesh->faces_end(); ++f_it) {
        final_face_count++;
    }
    std::cout << "QEM: Simplification complete." << std::endl;
    std::cout << "QEM: Initial faces: " << initial_num_faces << ", Target faces: " << target_num_faces
              << ", Final faces: " << final_face_count << " (Counted via iterator: " << current_num_faces << ")" << std::endl;
    if (iterations_count >= max_allowed_iterations) {
        std::cout << "QEM: Reached maximum allowed iterations (" << max_allowed_iterations << ")." << std::endl;
    }
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
    auto halfedge_mesh = operand_to_openmesh(&input_mesh);

    halfedge_mesh->request_vertex_status();
    halfedge_mesh->request_edge_status();
    halfedge_mesh->request_face_status();
    halfedge_mesh->request_halfedge_status();

    // QEM simplification
    qem(halfedge_mesh, simplification_ratio, distance_threshold);

    // Convert the simplified mesh back to the operand
    auto geometry = openmesh_to_operand(halfedge_mesh.get());

    auto mesh = geometry->get_component<MeshComponent>();

    // Set the output of the nodes
    params.set_output("Output", std::move(*geometry));

    return true;
}

NODE_DECLARATION_UI(qem);

NODE_DEF_CLOSE_SCOPE

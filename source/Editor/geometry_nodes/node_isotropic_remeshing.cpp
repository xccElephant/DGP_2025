#include <iostream>
#include <unordered_set>
#include <utility>

#include "GCore/Components/MeshOperand.h"
#include "GCore/util_openmesh_bind.h"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

// Helper function to compute vertex normal using area-weighted face normals
MyMesh::Normal compute_vertex_normal(
    std::shared_ptr<MyMesh> mesh,
    MyMesh::VertexHandle vh)
{
    MyMesh::Normal normal(0,0,0);
    float total_area = 0.0f;

    // Iterate through all faces adjacent to the vertex
    for(auto vf_it = mesh->vf_iter(vh); vf_it.is_valid(); ++vf_it) {
        auto fh = *vf_it;
        // Skip deleted faces
        if(mesh->status(fh).deleted())
            continue;

        // Calculate face normal and area
        MyMesh::Normal face_normal = mesh->calc_face_normal(fh);
        float face_area = mesh->calc_face_area(fh);

        // Area-weighted accumulation
        normal += face_normal * face_area;
        total_area += face_area;
    }

    if(total_area > 1e-6f) {
        normal /= total_area;
        if(normal.sqrnorm() > 1e-12f) {
            normal.normalize();
        } else {
            normal = MyMesh::Normal(0,0,1);
        }
    } else {
        // Case with no adjacent faces
        normal = MyMesh::Normal(0,0,1);
    }
    return normal;
}

void split_edges(std::shared_ptr<MyMesh> halfedge_mesh,float upper_bound)
{
    // Split all edges at their midpoint that are longer than upper_bound
    float upper_bound_sq = upper_bound * upper_bound;
    int split_count = 0;

    // Iterate through all edges
    for(MyMesh::EdgeIter eit = halfedge_mesh->edges_begin();
         eit != halfedge_mesh->edges_end();
         ++eit) {
        MyMesh::EdgeHandle eh = *eit;
        // Skip deleted edges
        if(halfedge_mesh->status(eh).deleted())
            continue;

        // Get the two vertices of the edge
        MyMesh::VertexHandle vh1 = halfedge_mesh->to_vertex_handle(
            halfedge_mesh->halfedge_handle(eh,0));
        MyMesh::VertexHandle vh2 = halfedge_mesh->to_vertex_handle(
            halfedge_mesh->halfedge_handle(eh,1));

        // Ensure vertices are valid
        if(halfedge_mesh->status(vh1).deleted() ||
            halfedge_mesh->status(vh2).deleted())
            continue;

        float edge_length_sq =
            (halfedge_mesh->point(vh1) - halfedge_mesh->point(vh2)).sqrnorm();

        // If edge length is greater than upper bound, split it
        if(edge_length_sq > upper_bound_sq) {
            MyMesh::Point midpoint =
                (halfedge_mesh->point(vh1) + halfedge_mesh->point(vh2)) * 0.5f;
            MyMesh::VertexHandle new_vh = halfedge_mesh->add_vertex(midpoint);
            halfedge_mesh->split(eh,new_vh);
            split_count++;
        }
    }

    // Perform garbage collection if any splits occurred
    if(split_count > 0) {
        halfedge_mesh->garbage_collection();
    }
}

void collapse_edges(std::shared_ptr<MyMesh> halfedge_mesh,float lower_bound)
{
    // Collapse all edges shorter than lower_bound into their midpoint
    float lower_bound_sq = lower_bound * lower_bound;
    int collapse_count = 0;

    // Iterate through all edges
    for(MyMesh::EdgeIter eit = halfedge_mesh->edges_begin();
         eit != halfedge_mesh->edges_end();
         ++eit) {
        MyMesh::EdgeHandle eh = *eit;
        // Skip deleted edges
        if(halfedge_mesh->status(eh).deleted())
            continue;

        // Get the two vertices of the edge
        MyMesh::VertexHandle vh1 = halfedge_mesh->to_vertex_handle(
            halfedge_mesh->halfedge_handle(eh,0));
        MyMesh::VertexHandle vh2 = halfedge_mesh->to_vertex_handle(
            halfedge_mesh->halfedge_handle(eh,1));

        // Ensure vertices are valid
        if(halfedge_mesh->status(vh1).deleted() ||
            halfedge_mesh->status(vh2).deleted())
            continue;

        float edge_length_sq =
            (halfedge_mesh->point(vh1) - halfedge_mesh->point(vh2)).sqrnorm();

        // If edge length is smaller than lower bound and collapse is valid, collapse it
        if(edge_length_sq < lower_bound_sq &&
            halfedge_mesh->is_collapse_ok(
            halfedge_mesh->halfedge_handle(eh,0))) {
            MyMesh::Point midpoint =
                (halfedge_mesh->point(vh1) + halfedge_mesh->point(vh2)) * 0.5f;
            halfedge_mesh->point(vh1) = midpoint;
            halfedge_mesh->collapse(halfedge_mesh->halfedge_handle(eh,0));
            collapse_count++;
        }
    }

    // Perform garbage collection if any collapses occurred
    if(collapse_count > 0) {
        halfedge_mesh->garbage_collection();
    }
}

void flip_edges(std::shared_ptr<MyMesh> halfedge_mesh)
{
    // Flip edges in order to minimize the deviation from valence 6 (or 4 on boundaries)

    // Lambda function to determine target valence: 6 for interior vertices, 4 for boundary vertices
    auto target_valence = [halfedge_mesh](MyMesh::VertexHandle vh) -> int {
        return halfedge_mesh->is_boundary(vh) ? 4 : 6;
    };

    // Lambda function to calculate valence deviation
    auto valence_deviation = [&target_valence](
                                 MyMesh::VertexHandle vh,int valence) -> int {
        return std::abs(valence - target_valence(vh));
    };

    int flip_count = 0;

    // Iterate through all edges
    for(MyMesh::EdgeIter eit = halfedge_mesh->edges_begin();
         eit != halfedge_mesh->edges_end();
         ++eit) {
        MyMesh::EdgeHandle eh = *eit;

        // Skip boundary edges and deleted edges
        if(halfedge_mesh->is_boundary(eh) ||
            halfedge_mesh->status(eh).deleted())
            continue;

        // Check if edge flip is valid
        if(!halfedge_mesh->is_flip_ok(eh))
            continue;

        auto heh = halfedge_mesh->halfedge_handle(eh,0);
        auto heh_opp = halfedge_mesh->opposite_halfedge_handle(heh);

        // Get the four vertices involved in the flip
        auto v0 = halfedge_mesh->from_vertex_handle(heh);
        auto v1 = halfedge_mesh->to_vertex_handle(heh);
        auto v2 = halfedge_mesh->to_vertex_handle(
            halfedge_mesh->next_halfedge_handle(heh));
        auto v3 = halfedge_mesh->to_vertex_handle(
            halfedge_mesh->next_halfedge_handle(heh_opp));

        // Skip if any vertex is deleted
        if(halfedge_mesh->status(v0).deleted() ||
            halfedge_mesh->status(v1).deleted() ||
            halfedge_mesh->status(v2).deleted() ||
            halfedge_mesh->status(v3).deleted())
            continue;

        // Calculate current valences
        int val0 = halfedge_mesh->valence(v0);
        int val1 = halfedge_mesh->valence(v1);
        int val2 = halfedge_mesh->valence(v2);
        int val3 = halfedge_mesh->valence(v3);

        // Calculate deviation before flip
        int deviation_before =
            valence_deviation(v0,val0) + valence_deviation(v1,val1) +
            valence_deviation(v2,val2) + valence_deviation(v3,val3);

        // Calculate deviation after flip (v0 and v1 lose 1, v2 and v3 gain 1)
        int deviation_after =
            valence_deviation(v0,val0 - 1) + valence_deviation(v1,val1 - 1) +
            valence_deviation(v2,val2 + 1) + valence_deviation(v3,val3 + 1);

        // Flip edge if it reduces total deviation
        if(deviation_after < deviation_before) {
            halfedge_mesh->flip(eh);
            flip_count++;
        }
    }

    // Perform garbage collection if any flips occurred
    if(flip_count > 0) {
        halfedge_mesh->garbage_collection();
    }
}

void relocate_vertices(std::shared_ptr<MyMesh> halfedge_mesh,float lambda)
{
    // Relocate vertices towards their gravity-weighted centroid

    // Ensure face normals are up to date
    halfedge_mesh->update_face_normals();
    int relocated_count = 0;

    // Iterate through all vertices
    for(MyMesh::VertexIter v_it = halfedge_mesh->vertices_begin();
         v_it != halfedge_mesh->vertices_end();
         ++v_it) {
        MyMesh::VertexHandle vh = *v_it;

        // Skip deleted vertices and boundary vertices
        if(halfedge_mesh->status(vh).deleted() ||
            halfedge_mesh->is_boundary(vh)) {
            continue;
        }

        MyMesh::Point current_pos = halfedge_mesh->point(vh);
        MyMesh::Normal normal = compute_vertex_normal(halfedge_mesh,vh);

        MyMesh::Point weighted_centroid(0,0,0);
        float total_weight = 0.0f;

        // Calculate gravity-weighted centroid using neighboring vertices
        for(MyMesh::VertexVertexIter vv_it = halfedge_mesh->vv_iter(vh);
             vv_it.is_valid(); ++vv_it) {
            MyMesh::VertexHandle neighbor_vh = *vv_it;

            // Skip deleted vertices
            if(halfedge_mesh->status(neighbor_vh).deleted())
                continue;

            MyMesh::Point neighbor_pos = halfedge_mesh->point(neighbor_vh);
            float weight = 0.0f;

            // Calculate weight as sum of areas of adjacent faces
            auto heh = halfedge_mesh->find_halfedge(vh,neighbor_vh);
            if(heh.is_valid()) {
                auto fh1 = halfedge_mesh->face_handle(heh);
                if(fh1.is_valid() && !halfedge_mesh->status(fh1).deleted()) {
                    weight += halfedge_mesh->calc_face_area(fh1);
                }
                auto heh_opp = halfedge_mesh->opposite_halfedge_handle(heh);
                auto fh2 = halfedge_mesh->face_handle(heh_opp);
                if(fh2.is_valid() && !halfedge_mesh->status(fh2).deleted()) {
                    weight += halfedge_mesh->calc_face_area(fh2);
                }
            }

            if(weight > 0) {
                weighted_centroid += neighbor_pos * weight;
                total_weight += weight;
            }
        }

        if(total_weight > 1e-6f) {
            relocated_count++;

            // Calculate gravity-weighted centroid
            MyMesh::Point g_i = weighted_centroid / total_weight;

            // Calculate direction from current position to centroid
            MyMesh::Point direction = g_i - current_pos;

            // Project direction onto tangent plane (remove normal component)
            float dot_product = direction | normal;
            MyMesh::Point projected_direction = direction - dot_product * normal;

            // Move vertex by lambda factor in projected direction
            MyMesh::Point new_position = current_pos + lambda * projected_direction;
            halfedge_mesh->point(vh) = new_position;
        }
    }
}

void isotropic_remeshing(
    std::shared_ptr<MyMesh> halfedge_mesh,
    float target_edge_length,
    int num_iterations,
    float lambda)
{
    for(int i = 0; i < num_iterations; ++i) {
        split_edges(halfedge_mesh,target_edge_length * 4.0f / 3.0f);
        collapse_edges(halfedge_mesh,target_edge_length * 4.0f / 5.0f);
        flip_edges(halfedge_mesh);
        relocate_vertices(halfedge_mesh,lambda);
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
    if(!mesh) {
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
    if(target_edge_length <= 0.0f) {
        std::cerr << "Isotropic Remeshing Node: Target edge length must be "
            "greater than zero."
            << std::endl;
        return false;
    }

    if(num_iterations < 0) {
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
        halfedge_mesh,target_edge_length,num_iterations,lambda);

    // Convert the remeshed halfedge mesh back to Geometry
    auto remeshed_geometry = openmesh_to_operand_trimesh(halfedge_mesh.get());
    // Set the output of the node
    params.set_output("Remeshed Mesh",std::move(*remeshed_geometry));

    return true;
}

NODE_DECLARATION_UI(isotropic_remeshing);
NODE_DEF_CLOSE_SCOPE

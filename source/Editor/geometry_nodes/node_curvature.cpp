#include <pxr/base/vt/array.h>

#include <vector>

#include "GCore/Components/MeshOperand.h"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

void compute_mean_curvature(
    const MyMesh& omesh,
    pxr::VtArray<float>& mean_curvature)
{
    // Initialize the array to store mean curvature for each vertex
    mean_curvature.resize(omesh.n_vertices());

    // Request vertex normals if they're not computed yet
    if(!omesh.has_vertex_normals())
        const_cast<MyMesh&>(omesh).request_vertex_normals();

    // Update normals
    const_cast<MyMesh&>(omesh).update_normals();

    // For each vertex in the mesh
    for(MyMesh::ConstVertexIter v_it = omesh.vertices_begin(); v_it != omesh.vertices_end(); ++v_it) {
        // Initialize Laplace-Beltrami operator
        OpenMesh::Vec3f laplace(0.0f,0.0f,0.0f);
        float weight_sum = 0.0f;

        // Vertex position
        OpenMesh::Vec3f vertex_position = omesh.point(*v_it);

        // For each vertex in the 1-ring neighborhood
        for(MyMesh::ConstVertexVertexIter vv_it = omesh.cvv_iter(*v_it); vv_it.is_valid(); ++vv_it) {
            // Get the position of the neighbor
            OpenMesh::Vec3f neighbor_position = omesh.point(*vv_it);

            // Get the half-edges associated with this edge
            MyMesh::HalfedgeHandle he = omesh.find_halfedge(*v_it,*vv_it);
            if(!he.is_valid())
                continue;

            // Get the opposite half-edge
            MyMesh::HalfedgeHandle he_opp = omesh.opposite_halfedge_handle(he);

            // Calculate cotangent weights
            float cot_alpha = 0.0f,cot_beta = 0.0f;

            // Get the faces for the cotangent calculation
            if(!omesh.is_boundary(he)) {
                // Get face vertices
                MyMesh::FaceHandle fh = omesh.face_handle(he);
                MyMesh::HalfedgeHandle he_next = omesh.next_halfedge_handle(he);
                MyMesh::VertexHandle vh_opposite = omesh.to_vertex_handle(he_next);

                // Calculate vectors
                OpenMesh::Vec3f vec1 = omesh.point(vh_opposite) - vertex_position;
                OpenMesh::Vec3f vec2 = omesh.point(vh_opposite) - neighbor_position;

                // Calculate cotangent using dot and cross product
                float dot_product = vec1.dot(vec2);
                float cross_length = (vec1 % vec2).length();

                if(cross_length > 1e-8)  // Avoid division by zero
                    cot_alpha = dot_product / cross_length;
            }

            if(!omesh.is_boundary(he_opp)) {
                // Get opposite face vertices
                MyMesh::FaceHandle fh_opp = omesh.face_handle(he_opp);
                MyMesh::HalfedgeHandle he_opp_next = omesh.next_halfedge_handle(he_opp);
                MyMesh::VertexHandle vh_opposite_opp = omesh.to_vertex_handle(he_opp_next);

                // Calculate vectors
                OpenMesh::Vec3f vec1 = omesh.point(vh_opposite_opp) - vertex_position;
                OpenMesh::Vec3f vec2 = omesh.point(vh_opposite_opp) - neighbor_position;

                // Calculate cotangent using dot and cross product
                float dot_product = vec1.dot(vec2);
                float cross_length = (vec1 % vec2).length();

                if(cross_length > 1e-8)  // Avoid division by zero
                    cot_beta = dot_product / cross_length;
            }

            // Total weight for this edge
            float weight = 0.5f * (cot_alpha + cot_beta);

            // Calculate the weighted offset vector
            laplace += weight * (neighbor_position - vertex_position);
            weight_sum += weight;
        }

        // Normalize by vertex area (approximated as 1/3 of the sum of adjacent face areas)
        float vertex_area = 0.0f;
        for(MyMesh::ConstVertexFaceIter vf_it = omesh.cvf_iter(*v_it); vf_it.is_valid(); ++vf_it) {
            MyMesh::ConstFaceVertexIter fv_it = omesh.cfv_iter(*vf_it);
            OpenMesh::Vec3f p0 = omesh.point(*fv_it);
            ++fv_it;
            OpenMesh::Vec3f p1 = omesh.point(*fv_it);
            ++fv_it;
            OpenMesh::Vec3f p2 = omesh.point(*fv_it);

            // Calculate face area using cross product
            vertex_area += ((p1 - p0) % (p2 - p0)).length() / 6.0f;  // 1/3 of the face area
        }

        // Calculate mean curvature as half the L2-norm of the Laplace-Beltrami operator
        if(vertex_area > 1e-8)  // Avoid division by zero
            laplace /= vertex_area;

        // Mean curvature is half the magnitude of the Laplace-Beltrami approximation
        mean_curvature[v_it->idx()] = 0.5f * laplace.length();

        // Determine sign using dot product with normal
        if(laplace.dot(omesh.normal(*v_it)) > 0)
            mean_curvature[v_it->idx()] *= -1.0f;
    }
}

void compute_gaussian_curvature(
    const MyMesh& omesh,
    pxr::VtArray<float>& gaussian_curvature)
{
    // Initialize the array to store Gaussian curvature for each vertex
    gaussian_curvature.resize(omesh.n_vertices());

    // Compute Gaussian curvature based on the Gauss-Bonnet theorem
    for(MyMesh::ConstVertexIter v_it = omesh.vertices_begin(); v_it != omesh.vertices_end(); ++v_it) {
        // For interior vertices, the angle sum should be 2π
        // For boundary vertices, the angle sum should be π
        float angle_deficit = (omesh.is_boundary(*v_it)) ? M_PI : (2.0f * M_PI);

        // Calculate vertex area (needed for the discrete Gaussian curvature calculation)
        float vertex_area = 0.0f;

        // Iterate through all incident faces
        for(MyMesh::ConstVertexFaceIter vf_it = omesh.cvf_iter(*v_it); vf_it.is_valid(); ++vf_it) {
            // Get the three vertices of the face
            MyMesh::ConstFaceVertexIter fv_it = omesh.cfv_iter(*vf_it);

            // Find the current vertex and its two neighbors in this face
            OpenMesh::VertexHandle current_vertex = *v_it;
            OpenMesh::VertexHandle v1,v2;

            // Find the other two vertices of the face
            bool found_first = false;

            for(; fv_it.is_valid(); ++fv_it) {
                if(*fv_it == current_vertex) {
                    continue;
                }

                if(!found_first) {
                    v1 = *fv_it;
                    found_first = true;
                } else {
                    v2 = *fv_it;
                    break;
                }
            }

            // Get positions of the vertices
            OpenMesh::Vec3f p0 = omesh.point(current_vertex);
            OpenMesh::Vec3f p1 = omesh.point(v1);
            OpenMesh::Vec3f p2 = omesh.point(v2);

            // Calculate vectors from the current vertex to the other vertices
            OpenMesh::Vec3f vec1 = p1 - p0;
            OpenMesh::Vec3f vec2 = p2 - p0;

            // Normalize vectors
            vec1.normalize();
            vec2.normalize();

            // Calculate the angle between these vectors using dot product
            float dot_product = std::max(-1.0f,std::min(1.0f,vec1.dot(vec2)));
            float angle = std::acos(dot_product);

            // Subtract this angle from the angle deficit
            angle_deficit -= angle;

            // Calculate the face area for vertex area computation
            float face_area = ((p1 - p0) % (p2 - p0)).length() / 2.0f;

            // Add one-third of the face area to the vertex area
            vertex_area += face_area / 3.0f;
        }

        // Calculate the Gaussian curvature as the angle deficit divided by the vertex area
        if(vertex_area > 1e-8)  // Avoid division by zero
            gaussian_curvature[v_it->idx()] = angle_deficit / vertex_area;
        else
            gaussian_curvature[v_it->idx()] = 0.0f;
    }
}

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(mean_curvature)
{
    b.add_input<Geometry>("Mesh");
    b.add_output<pxr::VtArray<float>>("Mean Curvature");
}

NODE_EXECUTION_FUNCTION(mean_curvature)
{
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_indices = mesh->get_face_vertex_indices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();

    // Convert the mesh to OpenMesh
    MyMesh omesh;

    // Add vertices
    std::vector<OpenMesh::VertexHandle> vhandles;
    vhandles.reserve(vertices.size());

    for(auto vertex : vertices) {
        omesh.add_vertex(OpenMesh::Vec3f(vertex[0],vertex[1],vertex[2]));
    }

    // Add faces
    size_t start = 0;
    for(int face_vertex_count : face_vertex_counts) {
        std::vector<OpenMesh::VertexHandle> face;
        face.reserve(face_vertex_count);
        for(int j = 0; j < face_vertex_count; j++) {
            face.push_back(
                OpenMesh::VertexHandle(face_vertex_indices[start + j]));
        }
        omesh.add_face(face);
        start += face_vertex_count;
    }

    // Compute mean curvature
    pxr::VtArray<float> mean_curvature;
    mean_curvature.reserve(omesh.n_vertices());

    compute_mean_curvature(omesh,mean_curvature);

    params.set_output("Mean Curvature",mean_curvature);

    return true;
}

NODE_DECLARATION_UI(mean_curvature);

NODE_DECLARATION_FUNCTION(gaussian_curvature)
{
    b.add_input<Geometry>("Mesh");
    b.add_output<pxr::VtArray<float>>("Gaussian Curvature");
}

NODE_EXECUTION_FUNCTION(gaussian_curvature)
{
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_indices = mesh->get_face_vertex_indices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();

    // Convert the mesh to OpenMesh
    MyMesh omesh;

    // Add vertices
    std::vector<OpenMesh::VertexHandle> vhandles;
    vhandles.reserve(vertices.size());

    for(auto vertex : vertices) {
        omesh.add_vertex(OpenMesh::Vec3f(vertex[0],vertex[1],vertex[2]));
    }

    // Add faces
    size_t start = 0;
    for(int face_vertex_count : face_vertex_counts) {
        std::vector<OpenMesh::VertexHandle> face;
        face.reserve(face_vertex_count);
        for(int j = 0; j < face_vertex_count; j++) {
            face.push_back(
                OpenMesh::VertexHandle(face_vertex_indices[start + j]));
        }
        omesh.add_face(face);
        start += face_vertex_count;
    }

    // Compute Gaussian curvature
    pxr::VtArray<float> gaussian_curvature;
    gaussian_curvature.reserve(omesh.n_vertices());

    compute_gaussian_curvature(omesh,gaussian_curvature);

    params.set_output("Gaussian Curvature",gaussian_curvature);

    return true;
}

NODE_DECLARATION_UI(gaussian_curvature);

NODE_DEF_CLOSE_SCOPE

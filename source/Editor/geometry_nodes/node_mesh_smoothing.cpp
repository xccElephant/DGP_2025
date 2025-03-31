#include <pxr/base/vt/array.h>

#include <cstdint>
#include <vector>
#include <set>

#include "GCore/Components/MeshOperand.h"
#include "GCore/GOP.h"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

typedef enum { kEdgeBased, kVertexBased } FaceNeighborType;

void getFaceArea(MyMesh &mesh, std::vector<float> &area)
{
    area.resize(mesh.n_faces());
    
    // Calculate area for each face
    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
        MyMesh::FaceVertexIter fv_it = mesh.fv_iter(*f_it);
        
        // Get the three vertices of the triangle
        MyMesh::Point v0 = mesh.point(*fv_it);
        ++fv_it;
        MyMesh::Point v1 = mesh.point(*fv_it);
        ++fv_it;
        MyMesh::Point v2 = mesh.point(*fv_it);
        
        // Calculate two edges
        MyMesh::Point edge1 = v1 - v0;
        MyMesh::Point edge2 = v2 - v0;
        
        // Calculate cross product and get area
        MyMesh::Point cross = edge1 % edge2;
        area[f_it->idx()] = 0.5f * cross.length();
    }
}

void getFaceCentroid(MyMesh &mesh, std::vector<MyMesh::Point> &centroid)
{
    centroid.resize(mesh.n_faces(), MyMesh::Point(0.0, 0.0, 0.0));
    
    // Calculate centroid for each face
    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
        MyMesh::Point sum(0.0, 0.0, 0.0);
        int count = 0;
        
        // Sum all vertices of the face
        for (MyMesh::FaceVertexIter fv_it = mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it) {
            sum += mesh.point(*fv_it);
            count++;
        }
        
        // Divide by the number of vertices to get centroid
        centroid[f_it->idx()] = sum / count;
    }
}

void getFaceNormal(MyMesh &mesh, std::vector<MyMesh::Normal> &normals)
{
    mesh.request_face_normals();
    mesh.update_face_normals();

    normals.resize(mesh.n_faces());
    
    // Get the normal for each face
    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
        normals[f_it->idx()] = mesh.normal(*f_it);
    }
}

void getFaceNeighbor(
    MyMesh &mesh,
    MyMesh::FaceHandle fh,
    std::vector<MyMesh::FaceHandle> &face_neighbor,
    FaceNeighborType neighbor_type = kEdgeBased)
{
    face_neighbor.clear();
    
    if (neighbor_type == kEdgeBased) {
        // Get face neighbors that share an edge with the current face
        for (MyMesh::FaceFaceIter ff_it = mesh.ff_iter(fh); ff_it.is_valid(); ++ff_it) {
            face_neighbor.push_back(*ff_it);
        }
    } else {
        // Get face neighbors that share at least one vertex with the current face
        std::set<MyMesh::FaceHandle> unique_neighbors;
        
        // Iterate through vertices of the face
        for (MyMesh::FaceVertexIter fv_it = mesh.fv_iter(fh); fv_it.is_valid(); ++fv_it) {
            // Iterate through faces adjacent to each vertex
            for (MyMesh::VertexFaceIter vf_it = mesh.vf_iter(*fv_it); vf_it.is_valid(); ++vf_it) {
                if (*vf_it != fh) {  // Don't include the face itself
                    unique_neighbors.insert(*vf_it);
                }
            }
        }
        
        // Convert set to vector
        for (const auto& neighbor : unique_neighbors) {
            face_neighbor.push_back(neighbor);
        }
    }
}

void getAllFaceNeighbor(
    MyMesh &mesh,
    std::vector<std::vector<MyMesh::FaceHandle>> &all_face_neighbor,
    bool include_central_face)
{
    all_face_neighbor.resize(mesh.n_faces());
    
    // Get neighbors for each face
    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
        std::vector<MyMesh::FaceHandle> neighbors;
        getFaceNeighbor(mesh, *f_it, neighbors);
        
        // Include the central face itself if requested
        if (include_central_face) {
            neighbors.push_back(*f_it);
        }
        
        all_face_neighbor[f_it->idx()] = neighbors;
    }
}

void updateVertexPosition(
    MyMesh &mesh,
    std::vector<MyMesh::Normal> &filtered_normals,
    int iteration_number,
    bool fixed_boundary)
{
    std::vector<MyMesh::Point> new_points(mesh.n_vertices());
    std::vector<MyMesh::Point> centroid;

    for (int iter = 0; iter < iteration_number; iter++) {
        getFaceCentroid(mesh, centroid);
        
        // Initialize new vertex positions with current positions
        for (MyMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
            new_points[v_it->idx()] = mesh.point(*v_it);
        }
        
        // Update non-boundary vertices
        for (MyMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
            // Skip boundary vertices if they should be fixed
            if (fixed_boundary && mesh.is_boundary(*v_it)) {
                continue;
            }
            
            // Calculate the new vertex position
            MyMesh::Point new_pos(0, 0, 0);
            float total_weight = 0.0f;
            
            // Iterate through adjacent faces
            for (MyMesh::VertexFaceIter vf_it = mesh.vf_iter(*v_it); vf_it.is_valid(); ++vf_it) {
                // Project the vertex onto the filtered normal plane passing through the face centroid
                MyMesh::Normal n = filtered_normals[vf_it->idx()];
                MyMesh::Point c = centroid[vf_it->idx()];
                
                float dist = (mesh.point(*v_it) - c) | n;  // Dot product for projection distance
                MyMesh::Point projected = mesh.point(*v_it) - n * dist;
                
                new_pos += projected;
                total_weight += 1.0f;
            }
            
            if (total_weight > 0) {
                new_points[v_it->idx()] = new_pos / total_weight;
            }
        }
        
        // Apply the new vertex positions
        for (MyMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
            mesh.set_point(*v_it, new_points[v_it->idx()]);
        }
    }
}

float getSigmaC(
    MyMesh &mesh,
    std::vector<MyMesh::Point> &face_centroid,
    float multiple_sigma_c)
{
    float sigma_c = 0.0;
    float num = 0.0;
    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end();
        f_it++) {
        MyMesh::Point ci = face_centroid[f_it->idx()];
        for (MyMesh::FaceFaceIter ff_it = mesh.ff_iter(*f_it); ff_it.is_valid();
            ff_it++) {
            MyMesh::Point cj = face_centroid[ff_it->idx()];
            sigma_c += (ci - cj).length();
            num++;
        }
    }
    sigma_c *= multiple_sigma_c / num;

    return sigma_c;
}

void update_filtered_normals_local_scheme(
    MyMesh &mesh,
    std::vector<MyMesh::Normal> &filtered_normals,
    float multiple_sigma_c,
    int normal_iteration_number,
    float sigma_s)
{
    filtered_normals.resize(mesh.n_faces());

    std::vector<std::vector<MyMesh::FaceHandle>> all_face_neighbor;
    getAllFaceNeighbor(mesh, all_face_neighbor, false);
    std::vector<MyMesh::Normal> previous_normals;
    getFaceNormal(mesh, previous_normals);
    std::vector<float> face_area;
    getFaceArea(mesh, face_area);
    std::vector<MyMesh::Point> face_centroid;
    getFaceCentroid(mesh, face_centroid);

    float sigma_c = getSigmaC(mesh, face_centroid, multiple_sigma_c);

    // Copy initial normals to filtered normals
    for (size_t i = 0; i < previous_normals.size(); i++) {
        filtered_normals[i] = previous_normals[i];
    }

    // Local and iterative scheme for filtering normals
    for (int iter = 0; iter < normal_iteration_number; iter++) {
        // Iterate over all faces
        for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
            MyMesh::FaceHandle fh = *f_it;
            int i = fh.idx();
            
            MyMesh::Normal weighted_normal(0, 0, 0);
            float total_weight = 0.0f;
            
            // Iterate over neighboring faces
            for (const auto& nfh : all_face_neighbor[i]) {
                int j = nfh.idx();
                
                // Calculate bilateral weights
                
                // Spatial weight (centroid distance)
                float distance = (face_centroid[i] - face_centroid[j]).length();
                float wc = exp(-distance*distance / (2.0f * sigma_c * sigma_c));
                
                // Signal weight (normal difference)
                float normal_difference = (1.0f - (filtered_normals[i] | filtered_normals[j]));
                float ws = exp(-normal_difference*normal_difference / (2.0f * sigma_s * sigma_s));
                
                // Combined weight
                float weight = wc * ws * face_area[j];
                
                // Add weighted normal
                weighted_normal += filtered_normals[j] * weight;
                total_weight += weight;
            }
            
            // Normalize the weighted normal
            if (total_weight > 0) {
                weighted_normal.normalize();
                previous_normals[i] = weighted_normal;
            }
        }
        
        // Update filtered normals for the next iteration
        for (size_t i = 0; i < filtered_normals.size(); i++) {
            filtered_normals[i] = previous_normals[i];
        }
    }
}

void bilateral_normal_filtering(
    MyMesh &mesh,
    float sigma_s,
    int normal_iteration_number,
    float multiple_sigma_c)
{
    std::vector<MyMesh::Normal> filtered_normals;
    update_filtered_normals_local_scheme(
        mesh,
        filtered_normals,
        multiple_sigma_c,
        normal_iteration_number,
        sigma_s);

    updateVertexPosition(mesh, filtered_normals, normal_iteration_number, true);
}

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(mesh_smoothing)
{
    b.add_input<Geometry>("Mesh");
    b.add_input<float>("Sigma_s").default_val(0.1).min(0).max(1);
    b.add_input<int>("Iterations").default_val(1).min(0).max(30);
    b.add_input<float>("Multiple Sigma C").default_val(1.0).min(0).max(10);

    b.add_output<Geometry>("Smoothed Mesh");
}

NODE_EXECUTION_FUNCTION(mesh_smoothing)
{
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();
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

    // Perform bilateral normal filtering
    float sigma_s = params.get_input<float>("Sigma_s");
    int iterations = params.get_input<int>("Iterations");
    float multiple_sigma_c = params.get_input<float>("Multiple Sigma C");

    bilateral_normal_filtering(omesh, sigma_s, iterations, multiple_sigma_c);

    // Convert back to Geometry
    pxr::VtArray<pxr::GfVec3f> smoothed_vertices;
    for (const auto &v : omesh.vertices()) {
        const auto &p = omesh.point(v);
        smoothed_vertices.push_back(pxr::GfVec3f(p[0], p[1], p[2]));
    }
    pxr::VtArray<int> smoothed_faceVertexIndices;
    pxr::VtArray<int> smoothed_faceVertexCounts;
    for (const auto &f : omesh.faces()) {
        size_t count = 0;
        for (const auto &vf : f.vertices()) {
            smoothed_faceVertexIndices.push_back(vf.idx());
            count += 1;
        }
        smoothed_faceVertexCounts.push_back(count);
    }

    Geometry smoothed_geometry;
    auto smoothed_mesh = std::make_shared<MeshComponent>(&smoothed_geometry);
    smoothed_mesh->set_vertices(smoothed_vertices);
    smoothed_mesh->set_face_vertex_indices(smoothed_faceVertexIndices);
    smoothed_mesh->set_face_vertex_counts(smoothed_faceVertexCounts);
    smoothed_geometry.attach_component(smoothed_mesh);
    params.set_output("Smoothed Mesh", smoothed_geometry);

    return true;
}

NODE_DECLARATION_UI(mesh_smoothing);

NODE_DEF_CLOSE_SCOPE

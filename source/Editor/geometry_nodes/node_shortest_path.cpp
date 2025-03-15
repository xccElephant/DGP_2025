#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <cstddef>
#include <string>
#include <queue>
#include <unordered_map>
#include <limits>

#include "GCore/Components/MeshOperand.h"
#include "GCore/GOP.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

/*
 * Find the shortest path between two vertices on a mesh using Dijkstra's algorithm.
 *
 * Inputs:
 *   - start_vertex_handle: Handle to the starting vertex
 *   - end_vertex_handle: Handle to the destination vertex
 *   - omesh: The mesh on which to find the path
 *
 * Outputs:
 *   - shortest_path_vertex_indices: List of vertex indices forming the shortest path
 *   - distance: Total length of the shortest path
 *
 * Returns:
 *   - true if a path exists between the vertices
 *   - false if no path exists
 */
bool find_shortest_path(
    const MyMesh::VertexHandle& start_vertex_handle,
    const MyMesh::VertexHandle& end_vertex_handle,
    const MyMesh& omesh,
    std::list<size_t>& shortest_path_vertex_indices,
    float& distance)
{
    // If start and end vertices are the same, return directly
    if(start_vertex_handle == end_vertex_handle) {
        shortest_path_vertex_indices.push_back(start_vertex_handle.idx());
        distance = 0.0f;
        return true;
    }

    // Map to store the shortest distance to each vertex
    std::unordered_map<size_t,float> dist;
    // Map to store the predecessor vertex for each vertex
    std::unordered_map<size_t,size_t> prev;
    // Priority queue: pair<distance, vertex index>
    std::priority_queue<std::pair<float,size_t>,
        std::vector<std::pair<float,size_t>>,
        std::greater<std::pair<float,size_t>>> pq;

    // Initialize distances
    for(auto vh : omesh.vertices()) {
        dist[vh.idx()] = std::numeric_limits<float>::infinity();
    }
    dist[start_vertex_handle.idx()] = 0.0f;

    // Add start vertex to priority queue
    pq.push(std::make_pair(0.0f,start_vertex_handle.idx()));

    // Dijkstra's algorithm main loop
    while(!pq.empty()) {
        auto current = pq.top();
        pq.pop();

        size_t current_idx = current.second;
        float current_dist = current.first;

        // If destination is found
        if(current_idx == end_vertex_handle.idx()) {
            break;
        }

        // Skip if a shorter path has already been found
        if(current_dist > dist[current_idx]) {
            continue;
        }

        // Iterate through all adjacent vertices
        MyMesh::VertexHandle current_vh(current_idx);
        for(auto voh_it = omesh.cvoh_iter(current_vh); voh_it.is_valid(); ++voh_it) {
            MyMesh::VertexHandle next_vh = omesh.to_vertex_handle(*voh_it);
            size_t next_idx = next_vh.idx();

            // Calculate edge length
            OpenMesh::Vec3f p1 = omesh.point(current_vh);
            OpenMesh::Vec3f p2 = omesh.point(next_vh);
            float edge_len = (p2 - p1).length();

            // Try to update distance
            float new_dist = dist[current_idx] + edge_len;
            if(new_dist < dist[next_idx]) {
                dist[next_idx] = new_dist;
                prev[next_idx] = current_idx;
                pq.push(std::make_pair(new_dist,next_idx));
            }
        }
    }

    // If no path was found
    if(dist[end_vertex_handle.idx()] == std::numeric_limits<float>::infinity()) {
        return false;
    }

    // Construct the shortest path
    distance = dist[end_vertex_handle.idx()];
    size_t current = end_vertex_handle.idx();
    while(current != start_vertex_handle.idx()) {
        shortest_path_vertex_indices.push_front(current);
        current = prev[current];
    }
    shortest_path_vertex_indices.push_front(start_vertex_handle.idx());

    return true;
}

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(shortest_path)
{
    b.add_input<std::string>("Picked Mesh [0] Name");
    b.add_input<std::string>("Picked Mesh [1] Name");
    b.add_input<Geometry>("Picked Mesh");
    b.add_input<size_t>("Picked Vertex [0] Index");
    b.add_input<size_t>("Picked Vertex [1] Index");

    b.add_output<std::list<size_t>>("Shortest Path Vertex Indices");
    b.add_output<float>("Shortest Path Distance");
}

NODE_EXECUTION_FUNCTION(shortest_path)
{
    auto picked_mesh_0_name =
        params.get_input<std::string>("Picked Mesh [0] Name");
    auto picked_mesh_1_name =
        params.get_input<std::string>("Picked Mesh [1] Name");
    // Ensure that the two picked meshes are the same
    if(picked_mesh_0_name != picked_mesh_1_name) {
        std::cerr << "Ensure that the two picked meshes are the same"
            << std::endl;
        return false;
    }

    auto mesh = params.get_input<Geometry>("Picked Mesh")
        .get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();
    auto face_vertex_indices = mesh->get_face_vertex_indices();

    // Convert the mesh to OpenMesh
    MyMesh omesh;

    // Add vertices
    std::vector<OpenMesh::VertexHandle> vhandles;
    vhandles.reserve(vertices.size());

    for (auto vertex : vertices) {
        omesh.add_vertex(OpenMesh::Vec3f(vertex[0], vertex[1], vertex[2]));
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

    auto start_vertex_index =
        params.get_input<size_t>("Picked Vertex [0] Index");
    auto end_vertex_index = params.get_input<size_t>("Picked Vertex [1] Index");

    // Turn the vertex indices into OpenMesh vertex handles
    OpenMesh::VertexHandle start_vertex_handle(start_vertex_index);
    OpenMesh::VertexHandle end_vertex_handle(end_vertex_index);

    // The indices of the vertices on the shortest path, including the start and
    // end vertices
    std::list<size_t> shortest_path_vertex_indices;

    // The distance of the shortest path
    float distance = 0.0f;

    if(find_shortest_path(
        start_vertex_handle,
        end_vertex_handle,
        omesh,
        shortest_path_vertex_indices,
        distance)) {
        params.set_output(
            "Shortest Path Vertex Indices",shortest_path_vertex_indices);
        params.set_output("Shortest Path Distance",distance);
        return true;
    } else {
        params.set_output("Shortest Path Vertex Indices",std::list<size_t>());
        params.set_output("Shortest Path Distance",0.0f);
        return false;
    }

    return true;
}

NODE_DECLARATION_UI(shortest_path);
NODE_DECLARATION_REQUIRED(shortest_path);

NODE_DEF_CLOSE_SCOPE

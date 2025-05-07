#include <functional>

#include "GCore/Components/MeshOperand.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(mvc)
{
    // The input is a 2D polygon on the XY plane
    b.add_input<Geometry>("Mesh");
    // The output is a function that takes the XY-coordinates of a point and
    // returns the mean value coordinates of that point with respect to the
    // input polygon
    b.add_output<std::function<std::vector<float>(float, float)>>(
        "Mean Value Coordinates");
}

NODE_EXECUTION_FUNCTION(mvc)
{
    // Get the input mesh
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();

    if (!mesh) {
        std::cerr
            << "MVC Node: Failed to get MeshComponent from input geometry."
            << std::endl;
        return false;
    }

    auto vertices = mesh->get_vertices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();
    auto face_vertex_indices = mesh->get_face_vertex_indices();

    // Ensure the input mesh is a 2D polygon
    if (vertices.size() < 3 || face_vertex_counts.size() != 1 ||
        face_vertex_counts[0] != vertices.size()) {
        std::cerr << "MVC Node: Input mesh must be a single polygon with at "
                     "least 3 vertices. "
                  << "Provided: " << vertices.size() << " vertices, "
                  << face_vertex_counts.size() << " faces. "
                  << "First face has "
                  << (face_vertex_counts.empty() ? 0 : face_vertex_counts[0])
                  << " vertices." << std::endl;

        return false;
    }

    // Ensure the polygon is on the XY plane
    for (const auto& vertex : vertices) {
        if (std::abs(vertex[2]) > 1e-5) {
            std::cerr << "MVC Node: Input mesh must be a 2D polygon on the XY "
                         "plane. Found vertex with Z-coordinate: "
                      << vertex[2] << std::endl;
            return false;
        }
    }

    // Extract the vertices of the polygon
    std::vector<std::array<float, 2>> polygon_vertices;
    for (int i = 0; i < face_vertex_counts[0]; i++) {
        auto vertex = vertices[face_vertex_indices[i]];
        polygon_vertices.push_back({ vertex[0], vertex[1] });
    }

    // Define the function that will compute mean value coordinates.
    // This lambda captures the polygon's vertices by value.
    auto mvc_function = [captured_vertices = polygon_vertices](
                            float p_x, float p_y) -> std::vector<float> {
        // TODO: Implement the Mean Value Coordinate (MVC) algorithm here.
        //
        // Input:
        //   - p_x, p_y: The XY-coordinates of the point for which to compute
        //     MVC.
        //
        //   - captured_vertices: A std::vector<std::array<float, 2>>
        //     representing the vertices of the 2D polygon. The polygon is on
        //     the XY plane. The order of vertices in this vector defines the
        //     polygon edges (e.g., v0-v1, v1-v2, ..., vN-1-v0).
        //
        // Output:
        //   - A std::vector<float> containing the mean value coordinates w_i
        //     for each vertex v_i of the polygon. The size of this vector
        //     must be equal to captured_vertices.size(). Each w_i corresponds
        //     to the vertex captured_vertices[i].
        //
        //   - The coordinates should sum to 1 (i.e., sum(w_i) = 1).
        //
        // Notes:
        //   - The polygon is assumed to be simple (non-self-intersecting).
        //   - Handle cases where the point (p_x, p_y) lies on an edge or
        //     coincides with a vertex.
        //
        // Algorithm Hint (Floater, 2003 - "Mean value coordinates"):
        //   For each vertex v_i = (x_i, y_i) of the polygon:
        //   1. Let r_i = ||v_i - p|| be the distance from p = (p_x, p_y) to
        //      v_i.
        //      If r_i is very small (p is close to v_i), then w_i = 1 and all
        //      other w_j = 0. Return.
        //   2. Let alpha_i be the angle of the triangle (p, v_i, v_{i+1}) at
        //      vertex p.
        //      (Indices are cyclic: v_N = v_0).
        //   3. The unnormalized weight for v_i is u_i = (tan(alpha_{i-1}/2) +
        //      tan(alpha_i/2)) / r_i.
        //      (alpha_{i-1} is the angle for triangle (p, v_{i-1}, v_i) at p).
        //   4. Compute all u_i for i = 0 to N-1.
        //   5. Normalize the weights: w_i = u_i / (sum of all u_j).
        //
        //   To compute tan(alpha/2):
        //   If alpha is the angle between two vectors A and B (emanating from
        //   p), tan(alpha/2) = ||A x B|| / (||A||*||B|| + A . B) for 3D
        //   vectors, or for 2D vectors A=(Ax,Ay), B=(Bx,By): tan(alpha/2) =
        //   (Ax*By - Ay*Bx) / (sqrt(Ax^2+Ay^2)*sqrt(Bx^2+By^2) + (Ax*Bx +
        //   Ay*By)) Alternatively, using lengths of triangle sides (p, v_i,
        //   v_{i+1}): let a = ||v_i - p||, b = ||v_{i+1} - p||, c = ||v_{i+1} -
        //   v_i||. By the law of cosines, cos(alpha_i) = (a^2 + b^2 - c^2) /
        //   (2ab). Then use half-angle identity: tan(alpha_i/2) = sqrt((1 -
        //   cos(alpha_i)) / (1 + cos(alpha_i))). Be careful with signs if
        //   alpha_i can be > PI (concave polygon relative to p). Floater's
        //   method is robust for arbitrary polygons.

        // Placeholder implementation:
        // This currently returns uniform weights, which is a valid set of
        // barycentric coordinates but not the Mean Value Coordinates.

        size_t num_vertices = captured_vertices.size();
        if (num_vertices == 0) {
            return {};  // Should not happen due to prior checks
        }
        // As a simple placeholder, return uniform weights.
        std::vector<float> uniform_weights(
            num_vertices, 1.0f / static_cast<float>(num_vertices));
        return uniform_weights;
    };

    // Set the output of the node
    params.set_output(
        "Mean Value Coordinates",
        std::function<std::vector<float>(float, float)>(mvc_function));
    return true;
}

NODE_DECLARATION_UI(mvc);

NODE_DEF_CLOSE_SCOPE

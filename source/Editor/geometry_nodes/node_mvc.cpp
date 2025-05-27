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
    b.add_output<std::function<std::vector<float>(float,float)>>(
        "Mean Value Coordinates");
}

NODE_EXECUTION_FUNCTION(mvc)
{
    // Get the input mesh
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();

    if(!mesh) {
        std::cerr
            << "MVC Node: Failed to get MeshComponent from input geometry."
            << std::endl;
        return false;
    }

    auto vertices = mesh->get_vertices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();
    auto face_vertex_indices = mesh->get_face_vertex_indices();

    // Ensure the input mesh is a 2D polygon
    if(vertices.size() < 3 || face_vertex_counts.size() != 1 ||
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
    for(const auto& vertex : vertices) {
        if(std::abs(vertex[2]) > 1e-5) {
            std::cerr << "MVC Node: Input mesh must be a 2D polygon on the XY "
                "plane. Found vertex with Z-coordinate: "
                << vertex[2] << std::endl;
            return false;
        }
    }

    // Extract the vertices of the polygon
    std::vector<std::array<float,2>> polygon_vertices;
    for(int i = 0; i < face_vertex_counts[0]; i++) {
        auto vertex = vertices[face_vertex_indices[i]];
        polygon_vertices.push_back({vertex[0],vertex[1]});
    }

    // Define the function that will compute mean value coordinates.
    // This lambda captures the polygon's vertices by value.
    auto mvc_function = [captured_vertices = polygon_vertices](
                            float p_x,float p_y) -> std::vector<float> {
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

        size_t n = captured_vertices.size();
        if(n == 0) {
            return {};
        }

        std::vector<float> weights(n);
        std::vector<float> r(n);      // Distances ||v_i - p||
        std::vector<float> tan_half_alpha(n); // tan(alpha_i / 2)

        constexpr float epsilon = 1e-9f; // Small epsilon for floating point comparisons

        // Calculate distances r_i and check if p coincides with a vertex
        for(size_t i = 0; i < n; ++i) {
            float dx = captured_vertices[i][0] - p_x;
            float dy = captured_vertices[i][1] - p_y;
            r[i] = std::sqrt(dx * dx + dy * dy);

            if(r[i] < epsilon) {
                // p coincides with vertex v_i
                std::fill(weights.begin(),weights.end(),0.0f);
                weights[i] = 1.0f;
                return weights;
            }
        }

        // Calculate tan(alpha_i / 2) for each triangle (p, v_i, v_{i+1})
        for(size_t i = 0; i < n; ++i) {
            size_t i_plus_1 = (i + 1) % n;

            // Vectors from p to v_i and p to v_{i+1}
            float v_i_px = captured_vertices[i][0] - p_x;
            float v_i_py = captured_vertices[i][1] - p_y;
            float v_i_plus_1_px = captured_vertices[i_plus_1][0] - p_x;
            float v_i_plus_1_py = captured_vertices[i_plus_1][1] - p_y;

            // r_i is ||v_i - p||, r_{i+1} is ||v_{i+1} - p||
            float r_i_val = r[i];
            float r_i_plus_1_val = r[i_plus_1];

            // Dot product: (v_i - p) . (v_{i+1} - p)
            float dot_product = v_i_px * v_i_plus_1_px + v_i_py * v_i_plus_1_py;
            // 2D cross product analog: (v_i - p) x (v_{i+1} - p) = (v_i_px * v_i_plus_1_py - v_i_py * v_i_plus_1_px)
            float cross_product_z = v_i_px * v_i_plus_1_py - v_i_py * v_i_plus_1_px;

            // Check for collinearity (p lies on the edge v_i v_{i+1})
            // If cross_product_z is close to 0, the points are collinear.
            // If dot_product < 0 and points are collinear, p is between v_i and v_{i+1}.
            // More robust check: distance from p to segment v_i v_{i+1} is near zero.
            // Let edge vector E = v_{i+1} - v_i
            float edge_x = captured_vertices[i_plus_1][0] - captured_vertices[i][0];
            float edge_y = captured_vertices[i_plus_1][1] - captured_vertices[i][1];
            float edge_len_sq = edge_x * edge_x + edge_y * edge_y;

            if(edge_len_sq < epsilon * epsilon) { // v_i and v_{i+1} are the same point, should not happen in a simple polygon
                tan_half_alpha[i] = 0; // Or handle error
                continue;
            }

            // Project (p - v_i) onto (v_{i+1} - v_i)
            // t = ((p - v_i) . (v_{i+1} - v_i)) / ||v_{i+1} - v_i||^2
            float t_numerator = (p_x - captured_vertices[i][0]) * edge_x + (p_y - captured_vertices[i][1]) * edge_y;
            float t = t_numerator / edge_len_sq;

            if(std::abs(cross_product_z) < epsilon && t > -epsilon && t < 1.0f + epsilon) {
                // p is on the segment v_i v_{i+1}
                // cross_product_z is small (collinear)
                // t is between 0 and 1 (p is between v_i and v_{i+1} or very close to them)
                std::fill(weights.begin(),weights.end(),0.0f);
                float dist_p_vi = r_i_val; // ||p - v_i||
                float dist_p_viplus1 = r_i_plus_1_val; // ||p - v_{i+1}||
                float edge_len = std::sqrt(edge_len_sq); // ||v_i - v_{i+1}||
                if(edge_len < epsilon) { // Should be caught by r[i] < epsilon if p is on a vertex
                    // This case means v_i and v_{i+1} are almost the same point,
                    // and p is on this point. Handled by vertex coincidence check.
                    // If somehow missed, assign based on which vertex p is closer to.
                    if(dist_p_vi < dist_p_viplus1) weights[i] = 1.0f; else weights[i_plus_1] = 1.0f;
                } else {
                    weights[i] = dist_p_viplus1 / edge_len;
                    weights[i_plus_1] = dist_p_vi / edge_len;
                }
                return weights;
            }

            // tan(alpha_i/2) = cross_product_z / (r_i * r_{i+1} + dot_product)
            // This is ||A x B|| / (||A||*||B|| + A . B) for 2D vectors if A and B emanate from origin
            // Here A = v_i - p, B = v_{i+1} - p
            // So ||A|| = r_i, ||B|| = r_{i+1}
            float denominator = r_i_val * r_i_plus_1_val + dot_product;
            if(std::abs(denominator) < epsilon) { // Avoid division by zero; angle is PI
                // This implies alpha_i is numerically PI.
                // This situation (p on segment v_i v_{i+1}) should ideally be caught by the
                // on-segment check which returns early. If we are here, it's an edge
                // case or extreme precision issue.
                // For alpha_i = PI, tan(alpha_i/2) approaches +infinity.
                // The cross_product_z should also be numerically zero if alpha_i is exactly PI.
                // Assigning +Inf seems most consistent with the limiting behavior
                // for weights when a point is on or very near an edge.
                tan_half_alpha[i] = std::numeric_limits<float>::infinity();
            } else {
                tan_half_alpha[i] = cross_product_z / denominator;
            }
        }

        // Calculate unnormalized weights u_i = (tan(alpha_{i-1}/2) + tan(alpha_i/2)) / r_i
        float sum_u = 0.0f;
        for(size_t i = 0; i < n; ++i) {
            size_t i_minus_1 = (i + n - 1) % n;
            float u_i = (tan_half_alpha[i_minus_1] + tan_half_alpha[i]) / r[i];
            weights[i] = u_i; // Store unnormalized weights first
            sum_u += u_i;
        }

        // Normalize weights: w_i = u_i / sum_u
        if(std::abs(sum_u) < epsilon) {
            // This can happen if p is far outside or symmetrically placed.
            // Or if all tan_half_alpha sums to 0 for each u_i.
            // Fallback to uniform weights or handle as an error/special case.
            // For points inside a simple polygon, sum_u should generally be non-zero.
            // std::cerr << "MVC Warning: Sum of unnormalized weights is close to zero." << std::endl;
            // Defaulting to uniform weights if sum_u is zero, though this indicates a potential issue or edge case.
            float uniform_weight = 1.0f / static_cast<float>(n);
            std::fill(weights.begin(),weights.end(),uniform_weight);
            return weights;
        }

        for(size_t i = 0; i < n; ++i) {
            weights[i] /= sum_u;
        }

        return weights;
    };

    // Set the output of the node
    params.set_output(
        "Mean Value Coordinates",
        std::function<std::vector<float>(float,float)>(mvc_function));
    return true;
}

NODE_DECLARATION_UI(mvc);

NODE_DEF_CLOSE_SCOPE

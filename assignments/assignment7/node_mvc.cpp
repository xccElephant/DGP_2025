#include <pxr/base/vt/array.h>

#include <functional>
#include <random>

#include "GCore/Components/MeshOperand.h"
#include "GCore/GOP.h"
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
        int num_vertices = captured_vertices.size();
        std::vector<float> distances(num_vertices);
        std::vector<float> tan_half(num_vertices);
        std::vector<float> weights(num_vertices);

        // Calculate distances from the point to each vertex
        for (int i = 0; i < num_vertices; ++i) {
            float x_i = captured_vertices[i][0];
            float y_i = captured_vertices[i][1];
            distances[i] = std::sqrt(
                (x_i - p_x) * (x_i - p_x) + (y_i - p_y) * (y_i - p_y));

            if (distances[i] < 1e-5) {
                // If the point is very close to a vertex, return 1 for that
                // vertex and 0 for all others.
                std::vector<float> weights(num_vertices, 0.0f);
                weights[i] = 1.0f;
                return weights;
            }
        }

        // Calculate the angles
        for (int i = 0; i < num_vertices; ++i) {
            float x_i = captured_vertices[i][0];
            float y_i = captured_vertices[i][1];
            int next_i = (i + 1) % num_vertices;
            float x_next = captured_vertices[next_i][0];
            float y_next = captured_vertices[next_i][1];
            float opposite_length = std::sqrt(
                (x_next - x_i) * (x_next - x_i) +
                (y_next - y_i) * (y_next - y_i));
            float cos_i = distances[i] * distances[i] +
                          distances[next_i] * distances[next_i] -
                          opposite_length * opposite_length;
            cos_i /= (2 * distances[i] * distances[next_i]);
            if (std::abs(cos_i - 1.0f) < 1e-5) {
                cos_i = 1.0f - 1e-5;
            }
            else if (std::abs(cos_i + 1.0f) < 1e-5) {
                cos_i = -1.0f + 1e-5;
            }
            tan_half[i] = std::sqrt((1 - cos_i) / (1 + cos_i));
        }

        // Calculate the weights
        for (int i = 0; i < num_vertices; ++i) {
            int prev_i = (i - 1 + num_vertices) % num_vertices;
            weights[i] = (tan_half[prev_i] + tan_half[i]) / distances[i];
        }

        // Normalize the weights
        float sum_weights = 0.0f;
        for (int i = 0; i < num_vertices; ++i) {
            sum_weights += weights[i];
        }
        for (int i = 0; i < num_vertices; ++i) {
            weights[i] /= sum_weights;
        }
        return weights;
    };

    // Set the output of the node
    params.set_output(
        "Mean Value Coordinates",
        std::function<std::vector<float>(float, float)>(mvc_function));
    return true;
}

NODE_DECLARATION_UI(mvc);

NODE_DEF_CLOSE_SCOPE

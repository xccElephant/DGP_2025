#include <iostream>
#include <string>

#include "nodes/core/def/node_def.hpp"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/structure.h"
#include "polyscope/surface_mesh.h"
#include "polyscope_widget/polyscope_renderer.h"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(get_control_points)
{
    b.add_input<std::string>("Strcture Name");
    b.add_output<std::vector<size_t>>("Control Points Indices");
    b.add_output<std::vector<std::array<float, 3>>>("Control Points Positions");
}

NODE_EXECUTION_FUNCTION(get_control_points)
{
    auto structure_name = params.get_input<std::string>("Strcture Name");
    structure_name = std::string(structure_name.c_str());

    bool is_empty = structure_name.empty();
    bool is_surface_mesh =
        polyscope::hasStructure("Surface Mesh", structure_name);
    bool is_curve_network =
        polyscope::hasStructure("Curve Network", structure_name);
    bool is_point_cloud =
        polyscope::hasStructure("Point Cloud", structure_name);

    if (is_empty ||
        (!is_surface_mesh && !is_curve_network && !is_point_cloud)) {
        std::cerr << "The structure is not found." << std::endl;
        return false;
    }

    std::vector<size_t> control_points =
        PolyscopeRenderer::GetControlPoints(structure_name);
    std::cout << "Control Points Indices: ";
    for (auto i : control_points) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    params.set_output("Control Points Indices", control_points);

    std::vector<std::array<float, 3>> control_points_positions;
    polyscope::Structure* structure;
    if (is_surface_mesh) {
        structure = polyscope::getStructure("Surface Mesh", structure_name);
        auto surface_mesh = static_cast<polyscope::SurfaceMesh*>(structure);
        for (auto index : control_points) {
            auto vertex = surface_mesh->vertexPositions.getValue(index);
            control_points_positions.push_back(
                { vertex[0], vertex[1], vertex[2] });
        }
    }
    else if (is_curve_network) {
        structure = polyscope::getStructure("Curve Network", structure_name);
        auto curve_network = static_cast<polyscope::CurveNetwork*>(structure);
        for (auto index : control_points) {
            auto vertex = curve_network->nodePositions.getValue(index);
            control_points_positions.push_back(
                { vertex[0], vertex[1], vertex[2] });
        }
    }
    else if (is_point_cloud) {
        structure = polyscope::getStructure("Point Cloud", structure_name);
        auto point_cloud = static_cast<polyscope::PointCloud*>(structure);
        for (auto index : control_points) {
            auto vertex = point_cloud->getPointPosition(index);
            control_points_positions.push_back(
                { vertex[0], vertex[1], vertex[2] });
        }
    }
    params.set_output("Control Points Positions", control_points_positions);

    return true;
}

NODE_DECLARATION_REQUIRED(get_control_points);
NODE_DECLARATION_UI(get_control_points);
NODE_DEF_CLOSE_SCOPE

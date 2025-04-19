#include "GCore/Components/MeshOperand.h"
#include "geom_node_base.h"
#include "GCore/util_openmesh_bind.h"
#include <Eigen/Sparse>
#include <vector>
#include <algorithm>

using Mesh = OpenMesh::PolyMesh_ArrayKernelT<>;

namespace utils {

// Basic component of parametrization
struct BaseTriangleFace {
    // Local rotation matrix L_t
    Eigen::Matrix2f Lt = Eigen::Matrix2f::Identity();
    // Local edge index
    std::array<int,3> local_idx = {-1,-1,-1};
    // Halfedge opp cot and local coordinate
    std::array<float,3> vert_cot = {0.f,0.f,0.f};
    std::array<Eigen::Vector2f,3> local_xy;
    // Original uv coordinate on unit square
    std::array<Eigen::Vector2f,3> global_uv;

    // Get local index by global index
    int local_index(int global_idx) const;
    // Updated uv coordinate
    void update_uv(const std::vector<Eigen::Vector2f>& uv_result);
    // Update local rotation matrix
    virtual void update_Lt() = 0;

    virtual ~BaseTriangleFace() = default;
};

// Get angle of p1-p0-p2
float compute_angle(
    const OpenMesh::Vec3f& p0,
    const OpenMesh::Vec3f& p1,
    const OpenMesh::Vec3f& p2);

// Get cotangent value of the opposite angle of halfedge he
float compute_opp_cotangent(
    const Mesh& mesh,
    const OpenMesh::HalfedgeHandle& he
);

float compute_area_2D(
    const Eigen::Vector2f& a,
    const Eigen::Vector2f& b,
    const Eigen::Vector2f& c);

float compute_area_3D(
    const OpenMesh::Vec3f& a,
    const OpenMesh::Vec3f& b,
    const OpenMesh::Vec3f& c);

pxr::VtArray<pxr::GfVec2f> normalize_uv_result(std::vector<Eigen::Vector2f>& uv);

std::vector<std::vector<OpenMesh::VertexHandle>> find_all_boundary_loops(
    const Mesh& mesh);

float calculate_boundary_length(
    const Mesh& mesh,
    const std::vector<OpenMesh::VertexHandle>& boundary_vertices);

std::vector<OpenMesh::VertexHandle> find_longest_boundary_loop(const Mesh& mesh);

// Convert two different data type
std::vector<Eigen::Vector2f> ConvertUSDToEigenUV(const pxr::VtArray<pxr::GfVec2f>& usdUV);
Eigen::Vector2f ConvertGfVec2ToEigen(const pxr::GfVec2f& v);

std::vector<Eigen::Vector3f> ConvertOpenMeshToEigenVertices(const Mesh& mesh);
Eigen::Vector3f ConvertVec3fToEigen(const OpenMesh::Vec3f& v);

pxr::VtArray<pxr::GfVec2f> ConvertEigenToUSDUV(const std::vector<Eigen::Vector2f>& eigenUV);
pxr::GfVec2f ConvertEigenToGfVec2(const Eigen::Vector2f& v);

} // namespace utils

namespace utils {

using namespace OpenMesh;

int BaseTriangleFace::local_index(int global_idx) const {
    for(int i = 0; i < 3; i++)
        if(local_idx[i] == global_idx) return i;
    return -1;
}

void BaseTriangleFace::update_uv(const std::vector<Eigen::Vector2f>& uv_result) {
    for(int i = 0; i < 3; i++)
        global_uv[i] = uv_result[local_idx[i]];
}

float compute_angle(const Vec3f& p0,const Vec3f& p1,const Vec3f& p2)
{
    const Vec3f e1 = (p1 - p0).normalized();
    const Vec3f e2 = (p2 - p0).normalized();

    const float dot_product = OpenMesh::dot(e1,e2);
    const float cos_theta = std::clamp(dot_product,-1.0f,1.0f);

    if(cos_theta >= 1.0f - 1e-6f) return 0.0f;
    if(cos_theta <= -1.0f + 1e-6f) return M_PI;
    return acos(cos_theta);
}

float compute_opp_cotangent(const Mesh& mesh,const HalfedgeHandle& he)
{
    if(!he.is_valid() || mesh.is_boundary(he)) return 0.0f;

    const auto v_from = mesh.from_vertex_handle(he);
    const auto v_to = mesh.to_vertex_handle(he);
    const auto v_next = mesh.to_vertex_handle(mesh.next_halfedge_handle(he));

    const float angle = compute_angle(
        mesh.point(v_next),
        mesh.point(v_from),
        mesh.point(v_to)
    );

    // Handle degenerated triangle
    if(angle < 1e-6f || angle > M_PI - 1e-6f) return 0.0f;
    return 1.0f / tan(angle);
}

float compute_area_2D(
    const Eigen::Vector2f& a,
    const Eigen::Vector2f& b,
    const Eigen::Vector2f& c)
{
    return 0.5f * std::abs(
        (b.x() - a.x()) * (c.y() - a.y()) -
        (b.y() - a.y()) * (c.x() - a.x())
    );
}

float compute_area_3D(
    const OpenMesh::Vec3f& a,
    const OpenMesh::Vec3f& b,
    const OpenMesh::Vec3f& c)
{
    OpenMesh::Vec3f edge1 = b - a;
    OpenMesh::Vec3f edge2 = c - a;
    return 0.5f * edge1.cross(edge2).norm();
}

pxr::VtArray<pxr::GfVec2f> normalize_uv_result(std::vector<Eigen::Vector2f>& uv) {
    // Get output uv from uv result
    size_t n_size = uv.size();

    pxr::VtArray<pxr::GfVec2f> uv_result(n_size);
    for(size_t i = 0; i < n_size; ++i) {
        uv_result[i] = utils::ConvertEigenToGfVec2(uv[i]);
    }

    // Obtain boundary value of all points
    float xmin = FLT_MAX,xmax = -FLT_MAX,ymin = FLT_MAX,ymax = -FLT_MAX;
    for(int i = 0; i < n_size; i++) {
        xmin = std::min(xmin,uv_result[i][0]);
        xmax = std::max(xmax,uv_result[i][0]);
        ymin = std::min(ymin,uv_result[i][1]);
        ymax = std::max(ymax,uv_result[i][1]);
    }

    // Normalize to unit square
    float scale = std::max(xmax - xmin,ymax - ymin);
    for(int i = 0; i < n_size; i++) {
        uv_result[i][0] = (uv_result[i][0] - xmin) / scale;
        uv_result[i][1] = (uv_result[i][1] - ymin) / scale;
    }

    return uv_result;
}

// Helper function to find all boundary loops in the mesh
std::vector<std::vector<VertexHandle>> find_all_boundary_loops(const Mesh& mesh)
{
    std::vector<std::vector<OpenMesh::VertexHandle>> boundary_loops;
    std::set<OpenMesh::VertexHandle> processed_vertices;

    for(auto he_it = mesh.halfedges_begin();
        he_it != mesh.halfedges_end();
        ++he_it)
    {
        if(mesh.is_boundary(*he_it))
        {
            OpenMesh::HalfedgeHandle start_he = *he_it;
            OpenMesh::VertexHandle start_vh = mesh.from_vertex_handle(start_he);

            // Skip if we've already processed this vertex as part of another boundary
            if(processed_vertices.find(start_vh) != processed_vertices.end())
                continue;

            // Trace this boundary loop
            std::vector<OpenMesh::VertexHandle> current_loop;
            OpenMesh::HalfedgeHandle current_he = start_he;
            do {
                OpenMesh::VertexHandle vh = mesh.from_vertex_handle(current_he);
                current_loop.push_back(vh);
                processed_vertices.insert(vh);
                current_he = mesh.next_halfedge_handle(current_he);
            } while(current_he != start_he);

            boundary_loops.push_back(current_loop);
        }
    }

    return boundary_loops;
}

// Helper function to calculate boundary length
float calculate_boundary_length(
    const Mesh& mesh,
    const std::vector<VertexHandle>& boundary_vertices)
{
    float total_length = 0.0f;
    for(size_t i = 0; i < boundary_vertices.size(); ++i)
    {
        auto v_curr = boundary_vertices[i];
        auto v_next = boundary_vertices[(i + 1) % boundary_vertices.size()];
        auto p_curr = mesh.point(v_curr);
        auto p_next = mesh.point(v_next);
        total_length += (p_next - p_curr).length();
    }
    return total_length;
}

std::vector<VertexHandle> find_longest_boundary_loop(const Mesh& mesh) {
    auto boundary_loops = utils::find_all_boundary_loops(mesh);
    if(boundary_loops.empty()) return {};

    auto longest_boundary = *std::max_element(
        boundary_loops.begin(),
        boundary_loops.end(),
        [&](const auto& a,const auto& b) {
        return utils::calculate_boundary_length(mesh,a) <
            utils::calculate_boundary_length(mesh,b);
    });

    return longest_boundary;
}

// === USD -> Eigen ===
std::vector<Eigen::Vector2f> ConvertUSDToEigenUV(const pxr::VtArray<pxr::GfVec2f>& usdUV)
{
    std::vector<Eigen::Vector2f> eigenUV;
    eigenUV.reserve(usdUV.size());
    for(const auto& uv : usdUV) {
        eigenUV.emplace_back(uv[0],uv[1]);
    }
    return eigenUV;
}

Eigen::Vector2f ConvertGfVec2ToEigen(const pxr::GfVec2f& v)
{
    return Eigen::Vector2f(v[0],v[1]);
}

// === OpenMesh -> Eigen ===
std::vector<Eigen::Vector3f> ConvertOpenMeshToEigenVertices(const Mesh& mesh)
{
    std::vector<Eigen::Vector3f> vertices;
    vertices.reserve(mesh.n_vertices());
    for(const auto& vh : mesh.vertices()) {
        const auto& p = mesh.point(vh);
        vertices.emplace_back(p[0],p[1],p[2]);
    }
    return vertices;
}

Eigen::Vector3f ConvertVec3fToEigen(const Vec3f& v)
{
    return Eigen::Vector3f(v[0],v[1],v[2]);
}

// === Eigen -> USD ===
pxr::VtArray<pxr::GfVec2f> ConvertEigenToUSDUV(const std::vector<Eigen::Vector2f>& eigenUV)
{
    pxr::VtArray<pxr::GfVec2f> usdUV;
    usdUV.reserve(eigenUV.size());
    for(const auto& uv : eigenUV) {
        usdUV.emplace_back(uv.x(),uv.y());
    }
    return usdUV;
}

pxr::GfVec2f ConvertEigenToGfVec2(const Eigen::Vector2f& v)
{
    return pxr::GfVec2f(v.x(),v.y());
}

}

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(circle_boundary_mapping)
{
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");
    // Output-1: Processed 3D mesh whose boundary is mapped to a square and the
    // interior vertices remains the same
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(circle_boundary_mapping)
{
    auto input = params.get_input<Geometry>("Input");
    if(!input.get_component<MeshComponent>())
        throw std::runtime_error("Boundary Mapping: Need Geometry Input.");

    auto mesh_ptr = operand_to_openmesh(&input);
    const Mesh& mesh = *mesh_ptr;

    // Find the longest boundary loop
    auto longest_boundary = utils::find_longest_boundary_loop(mesh);

    // Calculate total boundary length and cumulative lengths
    float total_length = utils::calculate_boundary_length(mesh,longest_boundary);
    std::vector<float> cumulative_lengths{0.0f};
    for(size_t i = 0; i < longest_boundary.size(); ++i)
    {
        auto v_curr = longest_boundary[i];
        auto v_next = longest_boundary[(i + 1) % longest_boundary.size()];
        auto p_curr = mesh.point(v_curr);
        auto p_next = mesh.point(v_next);
        float seg_length = (p_next - p_curr).length();
        cumulative_lengths.push_back(cumulative_lengths.back() + seg_length);
    }

    // Map vertices to unit circle
    for(size_t i = 0; i < longest_boundary.size(); ++i)
    {
        float t = cumulative_lengths[i] / total_length;
        float theta = 2.0 * M_PI * t;
        float x = 0.5 + 0.5 * cos(theta);
        float y = 0.5 + 0.5 * sin(theta);
        mesh_ptr->set_point(longest_boundary[i],OpenMesh::Vec3f(x,y,0.0f));
    }

    auto geometry = openmesh_to_operand(mesh_ptr.get());
    params.set_output("Output",std::move(*geometry));
    return true;
}

NODE_DECLARATION_FUNCTION(square_boundary_mapping)
{
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");

    // Output-1: Processed 3D mesh whose boundary is mapped to a square and the
    // interior vertices remains the same
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(square_boundary_mapping)
{
    auto input = params.get_input<Geometry>("Input");
    if(!input.get_component<MeshComponent>())
        throw std::runtime_error("Input does not contain a mesh");

    auto mesh_ptr = operand_to_openmesh(&input);
    const Mesh& mesh = *mesh_ptr;

    // Find the longest boundary loop
    auto longest_boundary = utils::find_longest_boundary_loop(mesh);

    // Calculate total boundary length and cumulative lengths
    float total_length = utils::calculate_boundary_length(mesh,longest_boundary);
    std::vector<float> cumulative_lengths{0.0f};
    for(size_t i = 0; i < longest_boundary.size(); ++i)
    {
        auto v_curr = longest_boundary[i];
        auto v_next = longest_boundary[(i + 1) % longest_boundary.size()];
        auto p_curr = mesh.point(v_curr);
        auto p_next = mesh.point(v_next);
        float seg_length = (p_next - p_curr).length();
        cumulative_lengths.push_back(cumulative_lengths.back() + seg_length);
    }

    // Map vertices to unit square
    for(size_t i = 0; i < longest_boundary.size(); ++i)
    {
        float t = cumulative_lengths[i] / total_length;
        float edge_param = t * 4.0;
        int edge_idx = static_cast<int>(edge_param) % 4;
        float local_t = edge_param - static_cast<int>(edge_param);

        float x = 0.0,y = 0.0;
        switch(edge_idx)
        {
        case 0: // Bottom edge
        x = local_t;
        y = 0.0;
        break;
        case 1: // Right edge
        x = 1.0;
        y = local_t;
        break;
        case 2: // Top edge
        x = 1.0 - local_t;
        y = 1.0;
        break;
        case 3: // Left edge
        x = 0.0;
        y = 1.0 - local_t;
        break;
        }
        mesh_ptr->set_point(longest_boundary[i],OpenMesh::Vec3f(x,y,0.0f));
    }

    auto geometry = openmesh_to_operand(mesh_ptr.get());
    params.set_output("Output",std::move(*geometry));
    return true;
}

NODE_DECLARATION_UI(boundary_mapping);
NODE_DEF_CLOSE_SCOPE

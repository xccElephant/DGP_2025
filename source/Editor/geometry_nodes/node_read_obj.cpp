
#include <pxr/base/vt/array.h>

#include "GCore/Components/MeshOperand.h"
#include "igl/readOBJ.h"
#include "nodes/core/def/node_def.hpp"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(read_obj_std)
{
    b.add_input<std::string>("Path").default_val("Default");
    b.add_output<std::vector<std::vector<float>>>("Vertices");
    b.add_output<std::vector<std::vector<float>>>("Texture Coordinates");
    b.add_output<std::vector<std::vector<float>>>("Normals");
    b.add_output<std::vector<std::vector<int>>>("Faces");
    b.add_output<std::vector<std::vector<int>>>("Face Texture Coordinates");
    b.add_output<std::vector<std::vector<int>>>("Face Normals");
    // Function content omitted
}

NODE_EXECUTION_FUNCTION(read_obj_std)
{
    auto path = params.get_input<std::string>("Path");
    std::vector<std::vector<float>> V;
    std::vector<std::vector<float>> TC;
    std::vector<std::vector<float>> N;
    std::vector<std::vector<int>> F;
    std::vector<std::vector<int>> FTC;
    std::vector<std::vector<int>> FN;
    // Function content omitted
    auto success = igl::readOBJ(path, V, TC, N, F, FTC, FN);

    if (success) {
        params.set_output("Vertices", std::move(V));
        params.set_output("Texture Coordinates", std::move(TC));
        params.set_output("Normals", std::move(N));
        params.set_output("Faces", std::move(F));
        params.set_output("Face Texture Coordinates", std::move(FTC));
        params.set_output("Face Normals", std::move(FN));
        return true;
    }
    else {
        return false;
    }
}

NODE_DECLARATION_UI(read_obj_std);

NODE_DECLARATION_FUNCTION(read_obj_eigen)
{
    b.add_input<std::string>("Path").default_val("Default");
    b.add_output<Eigen::MatrixXf>("Vertices");
    b.add_output<Eigen::MatrixXf>("Texture Coordinates");
    b.add_output<Eigen::MatrixXf>("Normals");
    b.add_output<Eigen::MatrixXi>("Faces");
    b.add_output<Eigen::MatrixXi>("Face Texture Coordinates");
    b.add_output<Eigen::MatrixXi>("Face Normals");
    // Function content omitted
}

NODE_EXECUTION_FUNCTION(read_obj_eigen)
{
    auto path = params.get_input<std::string>("Path");
    Eigen::MatrixXf V;
    Eigen::MatrixXf TC;
    Eigen::MatrixXf N;
    Eigen::MatrixXi F;
    Eigen::MatrixXi FTC;
    Eigen::MatrixXi FN;
    // Function content omitted
    auto success = igl::readOBJ(path, V, TC, N, F, FTC, FN);

    if (success) {
        params.set_output("Vertices", std::move(V));
        params.set_output("Texture Coordinates", std::move(TC));
        params.set_output("Normals", std::move(N));
        params.set_output("Faces", std::move(F));
        params.set_output("Face Texture Coordinates", std::move(FTC));
        params.set_output("Face Normals", std::move(FN));
        return true;
    }
    else {
        return false;
    }
}

NODE_DECLARATION_UI(read_obj_eigen);

NODE_DECLARATION_FUNCTION(read_obj_pxr)
{
    b.add_input<std::string>("Path").default_val("Default");
    b.add_output<pxr::VtVec3fArray>("Vertices");
    b.add_output<pxr::VtArray<int>>("FaceVertexCounts");
    b.add_output<pxr::VtArray<int>>("FaceVertexIndices");
    b.add_output<pxr::VtArray<pxr::GfVec3f>>("Normals");
    b.add_output<pxr::VtArray<pxr::GfVec2f>>("Texcoords");
    // Function content omitted
}

NODE_EXECUTION_FUNCTION(read_obj_pxr)
{
    const auto path = params.get_input<std::string>("Path");
    if (path.empty()) {
        return false;
    }

    Eigen::MatrixXf V, TC, N;
    Eigen::MatrixXi F, FTC, FN;

    if (!igl::readOBJ(path, V, TC, N, F, FTC, FN)) {
        return false;
    }

    pxr::VtVec3fArray vertices(V.rows());
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>>
        vertices_map(reinterpret_cast<float*>(vertices.data()), V.rows(), 3);
    vertices_map = V;

    const int num_faces = F.rows();
    const int vertices_per_face = F.cols();

    pxr::VtArray<int> faceVertexCounts(num_faces, vertices_per_face);

    pxr::VtArray<int> faceVertexIndices(num_faces * vertices_per_face);
    Eigen::Map<Eigen::VectorXi>(faceVertexIndices.data(), F.size()) =
        Eigen::Map<Eigen::VectorXi>(F.data(), F.size());

    pxr::VtArray<pxr::GfVec3f> normals;
    if (N.rows() > 0) {
        normals.resize(N.rows());
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>>
            normals_map(reinterpret_cast<float*>(normals.data()), N.rows(), 3);
        normals_map = N;
    }

    pxr::VtArray<pxr::GfVec2f> texcoords;
    if (TC.rows() > 0) {
        texcoords.resize(TC.rows());
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>>
            texcoords_map(
                reinterpret_cast<float*>(texcoords.data()), TC.rows(), 2);
        texcoords_map = TC.leftCols(2);
    }

    params.set_output("Vertices", std::move(vertices));
    params.set_output("FaceVertexCounts", std::move(faceVertexCounts));
    params.set_output("FaceVertexIndices", std::move(faceVertexIndices));
    params.set_output("Normals", std::move(normals));
    params.set_output("Texcoords", std::move(texcoords));

    return true;
}
NODE_DEF_CLOSE_SCOPE

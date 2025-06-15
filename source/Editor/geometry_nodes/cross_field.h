#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <complex>
#include <vector>

#include "Eigen/Eigen"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/SparseCore/SparseUtil.h"
#include "GCore/api.h"
#include "GCore/util_openmesh_bind.h"
#include "OpenMesh/Core/Geometry/EigenVectorT.hh"
#include "OpenMesh/Core/Mesh/Traits.hh"
#include "cmath"

// Define the traits for OpenMesh to use Eigen types
struct EigenTraits : OpenMesh::DefaultTraits {
    using Point = Eigen::Vector3d;
    using Normal = Eigen::Vector3d;
    using TexCoord2D = Eigen::Vector2d;
};

typedef OpenMesh::PolyMesh_ArrayKernelT<EigenTraits> EigenPolyMesh;

class CrossField {
   public:
    CrossField(EigenPolyMesh& _mesh);

    // Create constraints for the frame field
    // Since manually setting constraints is difficult,
    // we will generate constraints based on the mesh boundary
    void generateBoundaryConstrain();

    // Generate the corss field
    void generateCrossField();

    // Get the frame field for a specific face
    std::vector<std::vector<Eigen::Vector3d>> getCrossFields();
    std::vector<Eigen::Vector3d> getCrossFields(OpenMesh::FaceHandle fh);
    std::vector<Eigen::Vector3d> getCrossFields(int _fid);

   private:
    // Openmesh polymesh
    EigenPolyMesh* mesh;

    // Local coordinates for each face
    std::vector<std::array<Eigen::Vector3d, 2>> localCoords;

    // P(z) = z^4 - u^4

    // For each face, we store u
    std::vector<std::complex<double>> allComplex;

    // Constraints for the frame field
    std::unordered_map<OpenMesh::FaceHandle, std::complex<double>> constrains;

    // Generate local coordinates
    void generateLocalCoordinates();
};

typedef Eigen::SparseMatrix<std::complex<double>> SpMat;
typedef Eigen::Triplet<std::complex<double>> T;

CrossField::CrossField(EigenPolyMesh& _mesh) : mesh(&_mesh)
{
    allComplex.resize(mesh->n_faces());
    localCoords.resize(mesh->n_faces());

    mesh->request_vertex_status();
    mesh->request_edge_status();
    mesh->request_face_status();
    mesh->request_halfedge_status();
    mesh->request_face_normals();
    mesh->request_vertex_normals();

    generateLocalCoordinates();
}

void CrossField::generateBoundaryConstrain()
{
    // Set up constraints for the frame field
    // Prependicular to the boundary edges
    for (auto it = mesh->edges_sbegin(); it != mesh->edges_end(); it++) {
        if (!it->is_boundary()) {
            continue;
        }
        auto eh = *it;
        OpenMesh::SmartFaceHandle fh;
        if (eh.h0().face().is_valid())
            fh = eh.h0().face();
        else
            fh = eh.h1().face();
        // assert(fh.is_valid());
        if (!fh.is_valid()) {
            continue;
        }

        Eigen::Vector3d N = mesh->calc_face_normal(fh);
        Eigen::Vector3d p0 = mesh->point(eh.v0());
        Eigen::Vector3d p1 = mesh->point(eh.v1());
        Eigen::Vector3d dir = (p0 - p1).normalized();
        // Vector perpendicular to the boundary
        Eigen::Vector3d pp = dir.cross(N);

        // Set constraints
        double cos = pp.dot(localCoords[fh.idx()][0]);
        double sin = pp.dot(localCoords[fh.idx()][1]);

        // u = cos + i sin
        allComplex[fh.idx()] = std::complex<double>(cos, sin);
        constrains[fh] = allComplex[fh.idx()];
    }
}

void CrossField::generateCrossField()
{
    // TODO: Generate the cross field for the mesh

    /**
     * This method implements a least-squares optimization approach to create a
     * cross field that is smooth across the mesh while respecting user-defined
     * directional constraints.
     *
     * Cross Field Generation Algorithm:
     *     This algorithm generates a smooth 4-Rosy field over a
     *     triangular mesh surface using a least-squares optimization approach:
     *
     * 1. Input Validation
     *     - Verify the mesh is properly initialized and contains faces
     *
     * 2. Constraint System Construction
     *     - For each face and its neighboring faces connected by halfedges:
     *     - Compute the edge vector between adjacent face pairs
     *     - Project edge vectors onto local coordinate systems of both faces
     *     - Convert to complex numbers representing 4-fold rotational symmetry
     *       (4th power)
     *     - Build constraint equations: (e_f*)^4 = (e_g*)^4
     *
     * 3. Matrix Assembly
     *     - Construct rectangular matrix A where each row represents an edge
     *       constraint
     *     - Add boundary constraints as additional matrix rows
     *     - Build right-hand side vector b containing constraint values
     *
     * 4. Least-Squares Solution
     *     - Solve the overdetermined system AᵀAx = Aᵀb using LDLT
     *       decomposition
     *     - This minimizes the energy functional measuring field smoothness
     *
     * 5. Field Reconstruction
     *     - Extract 4th roots from the solution by dividing argument by 4
     *     - Store the resulting complex numbers representing the cross field
     *       orientation at each face
     *
     * The algorithm ensures the generated cross field is as smooth as possible
     * while satisfying boundary constraints, making it suitable for quad mesh
     * generation and texture synthesis applications.
     */
}

std::vector<std::vector<Eigen::Vector3d>> CrossField::getCrossFields()
{
    std::vector<std::vector<Eigen::Vector3d>> ret;
    for (int i = 0; i < mesh->n_faces(); i++) {
        ret.push_back(getCrossFields(i));
    }
    return ret;
}

std::vector<Eigen::Vector3d> CrossField::getCrossFields(OpenMesh::FaceHandle fh)
{
    return getCrossFields(fh.idx());
}

std::vector<Eigen::Vector3d> CrossField::getCrossFields(int _fid)
{
    std::vector<Eigen::Vector3d> ret;
    if (allComplex.size() <= _fid || _fid < 0)
        return ret;
    auto c = allComplex[_fid];
    auto& localCoord = localCoords[_fid];
    // 其中一条向量
    Eigen::Vector3d dir = c.real() * localCoord[0] + c.imag() * localCoord[1];
    auto N = mesh->calc_face_normal(mesh->face_handle(_fid));
    // 第一条向量
    ret.push_back(dir);
    for (int i = 0; i < 3; i++) {
        Eigen::Vector3d d = N.cross(dir);
        dir = d;
        ret.push_back(dir);
    }
    return ret;
}

void CrossField::generateLocalCoordinates()
{
    // TODO: Generate local coordinates for each face, fill in localCoords
}

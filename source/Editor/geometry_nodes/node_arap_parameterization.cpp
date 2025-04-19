#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
#include <memory>

#include "GCore/Components/MeshOperand.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"

#include <cmath>
#include <time.h>
#include <Eigen/Dense>
#include <Eigen/SparseLU>
#include <Eigen/Sparse>
#include <iostream>
#include <Eigen/Dense>
namespace USTC_CG
{
class AXAP
{
public:

    AXAP() = default;
    virtual ~AXAP() = default;
    virtual Eigen::Matrix2f localLinearFit() = 0;
    float cotangent(std::vector<float> a,std::vector<float> b);
    std::vector<float> cross(std::vector<float> a,std::vector<float> b);
    std::vector<float> difference(std::vector<float> a,std::vector<float> b);
    virtual Eigen::Matrix3f coefficientsFromFace(std::vector<std::vector<float>> triangle,
        std::vector<bool> is_border,std::vector<bool> is_fixed) = 0; //get the coefficients of the global equations
    virtual Eigen::MatrixXf bFromFace(Eigen::Matrix2f J,std::vector<std::vector<float>> triangle,
        std::vector<std::vector<float>>,
        std::vector<bool> is_border,std::vector<bool> is_fixed) = 0;
    virtual void output_x() {};
    virtual void output_y() {};
protected:

    //store the decomposition of global coefficient matrix
    std::vector<std::vector<float>> mesh_triangle_;
    std::vector<std::vector<float>> parameter_triangle_;

};
}

namespace USTC_CG
{
float AXAP::cotangent(std::vector<float> a,std::vector<float> b)
{
    //return a dot b

    if(a.size() != 3 || b.size() != 3)
        throw std::runtime_error("vector size error");
    float dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];


    //return the abs of a X b

    float cross_abs = sqrt(
        pow(a[1] * b[2] - a[2] * b[1],2) +
        pow(a[2] * b[0] - a[0] * b[2],2) +
        pow(a[0] * b[1] - a[1] * b[0],2));

    //return cotangent of the angle between a and b

    return dot / cross_abs;

}
std::vector<float> AXAP::cross(std::vector<float> a,std::vector<float> b)
{
    std::vector<float> cross;
    if(a.size() != 3 || b.size() != 3)
        throw std::runtime_error("vector size error");
    cross.push_back(a[1] * b[2] - a[2] * b[1]);
    cross.push_back(a[2] * b[0] - a[0] * b[2]);
    cross.push_back(a[0] * b[1] - a[1] * b[0]);
    return cross;
}
std::vector<float> AXAP::difference(std::vector<float> a,std::vector<float> b)
{
    std::vector<float> difference;
    if(a.size() != 3 || b.size() != 3)
        throw std::runtime_error("triangle error!");
    for(int i = 0; i < 3; i++)
    {
        difference.push_back(a[i] - b[i]);
    }
    return difference;

}
}

namespace USTC_CG
{
class ARAP: public AXAP
{
public:
    ARAP() = default;
    virtual ~ARAP() = default;
    ARAP(std::vector<std::vector<float>> mesh_triangle,
        std::vector<std::vector<float>> parameter_triangle)
    {
        mesh_triangle_ = mesh_triangle;
        parameter_triangle_ = parameter_triangle;
    };
    Eigen::Matrix2f localLinearFit() override;
    Eigen::Matrix3f coefficientsFromFace(std::vector<std::vector<float>> triangle,
        std::vector<bool> is_border,std::vector<bool> is_fixed) override;
    Eigen::MatrixXf bFromFace(Eigen::Matrix2f J,std::vector<std::vector<float>> triangle,
        std::vector<std::vector<float>> parameter_triangle,
        std::vector<bool> is_border,std::vector<bool> is_fixed) override;
    void output_x() override;
    void output_y() override;
private:
    Eigen::Vector3f x_axis_;
    Eigen::Vector3f y_axis_;
};
}

namespace USTC_CG
{

Eigen::Matrix2f ARAP::localLinearFit()
{
    //test
    Eigen::Matrix2f A;
    A.setZero();
    x_axis_.setZero();
    std::vector<float> x_axis_vec;
    x_axis_<<
        mesh_triangle_[1][0] - mesh_triangle_[2][0],
        mesh_triangle_[1][1] - mesh_triangle_[2][1],
        mesh_triangle_[1][2] - mesh_triangle_[2][2]; //chosen x_aixs of the triangle plane
    x_axis_ = x_axis_ / x_axis_.norm();

    y_axis_.setZero();

    std::vector<float> y_axis_vec;
    std::vector<float> z_axis_vec; //bridge to calculate y axis

    std::vector<float> coefficients;
    std::vector<float> v1,v2;

    for(int i = 0; i < 3; i++)
    {
        x_axis_vec.push_back(x_axis_(i));
    }

    z_axis_vec = cross(difference(mesh_triangle_[1],mesh_triangle_[2]),difference(mesh_triangle_[0],mesh_triangle_[2]));
    for(int i = 0; i < 3; i++)
    {
        z_axis_vec[i] = z_axis_vec[i] / sqrt(pow(z_axis_vec[0],2) + pow(z_axis_vec[1],2) + pow(z_axis_vec[2],2));
    }

    y_axis_vec = cross(z_axis_vec,x_axis_vec);

    for(int i = 0; i < 3; i++)
    {
        y_axis_(i) = y_axis_vec[i];
    }

    y_axis_ = y_axis_ / y_axis_.norm();

    for(int i = 0; i < 3; i++)
    {
        v1.push_back(mesh_triangle_[0][i] - mesh_triangle_[2][i]);
        v2.push_back(mesh_triangle_[1][i] - mesh_triangle_[2][i]);
    }
    coefficients.push_back(cotangent(v1,v2));

    v1.clear();
    v2.clear();
    for(int i = 0; i < 3; i++)
    {
        v1.push_back(mesh_triangle_[1][i] - mesh_triangle_[0][i]);
        v2.push_back(mesh_triangle_[2][i] - mesh_triangle_[0][i]);
    }
    coefficients.push_back(cotangent(v1,v2));

    v1.clear();
    v2.clear();
    for(int i = 0; i < 3; i++)
    {
        v1.push_back(mesh_triangle_[0][i] - mesh_triangle_[1][i]);
        v2.push_back(mesh_triangle_[2][i] - mesh_triangle_[1][i]);
    }
    coefficients.push_back(cotangent(v1,v2));

    v1.clear();
    v2.clear();

    //test
    for(int i = 0; i < 3; i++)
    {
        Eigen::Vector2f u(parameter_triangle_[i][0] - parameter_triangle_[(i + 1) % 3][0],
            parameter_triangle_[i][1] - parameter_triangle_[(i + 1) % 3][1]);
        Eigen::Vector3f x(mesh_triangle_[i][0] - mesh_triangle_[(i + 1) % 3][0],
            mesh_triangle_[i][1] - mesh_triangle_[(i + 1) % 3][1],
            mesh_triangle_[i][2] - mesh_triangle_[(i + 1) % 3][2]);
        Eigen::Vector2f x_p(x.dot(x_axis_),x.dot(y_axis_));
        A += coefficients[i] * u * x_p.transpose();
    }
    Eigen::BDCSVD<Eigen::Matrix2f> svd(A,Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2f U = svd.matrixU();
    Eigen::Matrix2f V = svd.matrixV();
    Eigen::VectorXf S = svd.singularValues();
    Eigen::Matrix2f R_test = V.transpose() * U;
    if(S(0) < 0)
    {
        U(0,0) = -U(0,0);
        U(1,0) = -U(1,0);
    }
    if(A.determinant() > 0)
    {
        if(S(1) < 0)
        {
            U(0,1) = -U(0,1);
            U(1,1) = -U(1,1);
        }
    } else
    {
        if(S(1) > 0)
        {
            U(0,1) = -U(0,1);
            U(1,1) = -U(1,1);
        }
    }
    Eigen::Matrix2f R = U * V.transpose();
    return R;
}

Eigen::Matrix3f ARAP::coefficientsFromFace(std::vector<std::vector<float>> triangle,
    std::vector<bool> is_border,std::vector<bool> is_fixed)
{

    if(triangle.size() != 3)
    {
        throw std::runtime_error("triangle error!");
    }

    for(int i = 0; i < 3; i++)
    {
        if(triangle[i].size() != 3)
        {

            throw std::runtime_error("triangle error!");
        }
    }
    Eigen::Matrix3f A_local;
    A_local.setZero();
    float a01 = cotangent(difference(triangle[0],triangle[2]),difference(triangle[1],triangle[2]));
    float a12 = cotangent(difference(triangle[1],triangle[0]),difference(triangle[2],triangle[0]));
    float a20 = cotangent(difference(triangle[2],triangle[1]),difference(triangle[0],triangle[1]));
    //inner half edge
    A_local(0,0) += a01;
    A_local(0,1) -= a01;
    A_local(1,1) += a12;
    A_local(1,2) -= a12;
    A_local(2,2) += a20;
    A_local(2,0) -= a20;
    //outer half edge concerning angles inside the triangle
    if(is_border[0] == 0)
    {
        A_local(1,1) += a01;
        A_local(1,0) -= a01;
    }
    if(is_border[1] == 0)
    {
        A_local(2,2) += a12;
        A_local(2,1) -= a12;
    }
    if(is_border[2] == 0)
    {
        A_local(0,0) += a20;
        A_local(0,2) -= a20;
    }

    if(is_fixed[0] == 1)
    {
        A_local(1,0) = 0;
        A_local(2,0) = 0;
    }
    if(is_fixed[1] == 1)
    {
        A_local(0,1) = 0;
        A_local(2,1) = 0;
    }
    if(is_fixed[2] == 1)
    {
        A_local(0,2) = 0;
        A_local(1,2) = 0;
    }
    return A_local;
}

Eigen::MatrixXf ARAP::bFromFace(Eigen::Matrix2f J,std::vector<std::vector<float>> triangle,
    std::vector<std::vector<float>> parameter_triangle,
    std::vector<bool> is_border,std::vector<bool> is_fixed)
{
    if(triangle.size() != 3)
        throw std::runtime_error("triangle error!");
    Eigen::MatrixXf B(3,2);
    B.setZero();
    float a01 = cotangent(difference(triangle[0],triangle[2]),difference(triangle[1],triangle[2]));
    float a12 = cotangent(difference(triangle[1],triangle[0]),difference(triangle[2],triangle[0]));
    float a20 = cotangent(difference(triangle[2],triangle[1]),difference(triangle[0],triangle[1]));
    Eigen::Vector3f x01,x12,x20;
    x01.setZero();
    x12.setZero();
    x20.setZero();
    Eigen::Vector2f x_p01,x_p12,x_p20;
    x_p01.setZero();
    x_p12.setZero();
    x_p20.setZero();

    for(int i = 0; i < 3; i++)
    {
        x01(i) = difference(triangle[0],triangle[1])[i];
        x12(i) = difference(triangle[1],triangle[2])[i];
        x20(i) = difference(triangle[2],triangle[0])[i];
    }
    x_p01<<x01.dot(x_axis_),x01.dot(y_axis_);
    x_p12<<x12.dot(x_axis_),x12.dot(y_axis_);
    x_p20<<x20.dot(x_axis_),x20.dot(y_axis_);

    for(int i = 0; i < 2; i++)
    {
        B(0,i) += (a01 * J * x_p01)(i);
        B(1,i) += (a12 * J * x_p12)(i);
        B(2,i) += (a20 * J * x_p20)(i);
        if(is_border[0] == 0)
        {
            B(1,i) -= (a01 * J * x_p01)(i);
            if(is_fixed[0] == 1)
            {
                B(1,i) += (parameter_triangle[0][i] - parameter_triangle[1][i]) * a01;
            }
        }
        if(is_border[1] == 0)
        {
            B(2,i) -= (a12 * J * x_p12)(i);
            if(is_fixed[1] == 1)
            {
                B(2,i) += (parameter_triangle[1][i] - parameter_triangle[2][i]) * a12;
            }
        }
        if(is_border[2] == 0)
        {
            B(0,i) -= (a20 * J * x_p20)(i);
            if(is_fixed[2] == 1)
            {
                B(0,i) += (parameter_triangle[2][i] - parameter_triangle[0][i]) * a20;
            }
        }
    }
    return B;
}
void ARAP::output_x()
{
    std::cout << "x_axis: " << x_axis_ << std::endl;
}
void ARAP::output_y()
{
    std::cout << "y_axis: " << y_axis_ << std::endl;
    std::cout << "x dot y: " << x_axis_.dot(y_axis_) << std::endl;
}
}

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;

void arap(
    std::shared_ptr<MyMesh> halfedge_mesh,
    std::shared_ptr<MyMesh> iter_mesh)
{
    // TODO: Implement ARAP Parameterization Algorithm.

    /* ------------- ARAP Parameterization Implementation
    -----------
     ** Implement ARAP mesh parameterization to minimize local distortion.
     ** Steps:
     ** 1. Initial Setup: Use a HW4 parameterization result as initial setup.
     ** 2. Local Phase: For each triangle, compute local orthogonal approximation
     **    (Lt) by computing SVD of Jacobian(Jt) with fixed u.
     ** 3. Global Phase: With Lt fixed, update parameter coordinates(u) by solving
     **    a pre-factored global sparse linear system.
     ** 4. Iteration: Repeat Steps 2 and 3 to refine parameterization.
     */
     // ARAP Parameterization Algorithm Implementation

     // Step 1: Use iter_mesh (from HW4) as initial parameterization
     // Copy the current parameterization from iter_mesh to ensure we're working with the initial embedding

     // Step 2-4: Implement ARAP iteration
}

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(arap_parameterization)
{
    // // Input-1: Original 3D mesh with boundary
    // b.add_input<Geometry>("Input");

    // // Input-2: An embedding result of the mesh. Use the XY coordinates of the
    // // embedding as the initialization of the ARAP algorithm
    // //
    // // Here we use **the result of Assignment 4** as the initialization
    // b.add_input<Geometry>("Initialization");

    // // Output-1: Like the result of Assignment 4, output the 2D embedding of the
    // // mesh
    // b.add_output<Geometry>("Output");

    // Input-1: Original 3D mesh with boundary
    // Maybe you need to add another input for initialization?
    b.add_input<Geometry>("Input");
    b.add_input<Geometry>("Initial guess");
    b.add_input<int>("iteration number").default_val(5).min(1).max(10);
    /*
    ** NOTE: You can add more inputs or outputs if necessary. For example, in
    ** some cases, additional information (e.g. other mesh geometry, other
    ** parameters) is required to perform the computation.
    **
    ** Be sure that the input/outputs do not share the same name. You can add
    ** one geometry as
    **
    **                b.add_input<Geometry>("Input");
    **
    ** Or maybe you need a value buffer like:
    **
    **                b.add_input<float1Buffer>("Weights");
    */

    // Output-1: The UV coordinate of the mesh, provided by ARAP algorithm
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(arap_parameterization)
{
    // Get the input from params
    // auto input = params.get_input<Geometry>("Input");
    // auto iters = params.get_input<Geometry>("Initialization");

    // // Avoid processing the node when there is no input
    // if (!input.get_component<MeshComponent>() ||
    //     !iters.get_component<MeshComponent>()) {
    //     std::cerr << "ARAP Parameterization: Need Geometry Input." << std::endl;
    // }

    // /* ----------------------------- Preprocess -------------------------------
    // ** Create a halfedge structure (using OpenMesh) for the input mesh. The
    // ** half-edge data structure is a widely used data structure in geometric
    // ** processing, offering convenient operations for traversing and modifying
    // ** mesh elements.
    // */

    // Initialization
    // auto halfedge_mesh = operand_to_openmesh(&input);
    // auto iter_mesh = operand_to_openmesh(&iters);

    // ARAP parameterization
    // arap(halfedge_mesh, iter_mesh);

    // auto geometry = openmesh_to_operand(iter_mesh.get());

    // Set the output of the nodes
    // params.set_output("Output", std::move(*geometry));
    // return true;
    // Get the input from params

    auto input = params.get_input<Geometry>("Input");
    auto initial_guess = params.get_input<Geometry>("Initial guess");
    int N = params.get_input<int>("iteration number");

    // Avoid processing the node when there is no input
    if(!input.get_component<MeshComponent>())
    {
        throw std::runtime_error("Need Geometry Input.");
    }
    /* ----------------------------- Preprocess -------------------------------
    ** Create a halfedge structure (using OpenMesh) for the input mesh. The
    ** half-edge data structure is a widely used data structure in geometric
    ** processing, offering convenient operations for traversing and modifying
    ** mesh elements.
    */
    auto halfedge_mesh = operand_to_openmesh(&input);
    auto parameter_mesh = operand_to_openmesh(&initial_guess);

    if(halfedge_mesh->n_vertices() > 1)
    {
        int n_points = halfedge_mesh->n_vertices();
        //global equation: AX = B
        Eigen::SparseMatrix<float> A(n_points,n_points);
        Eigen::MatrixXf B(n_points,2);
        Eigen::MatrixXf X(n_points,2);
        Eigen::MatrixXf temp(n_points,n_points); //save A temporarily
        Eigen::SparseLU<Eigen::SparseMatrix<float>> solver; //equation solver
        int fixed_vertex = 0;
        int reference_vertex = 1;
        bool set_reference = false;
        float fixed_distance = 0.0f;
        A.setZero();//initialize A only once
        for(int n = 0; n < N; n++)
        {
            int test = 0;
            std::vector<int> fv_idx; //idx of vertices on a face
            std::vector<std::vector<float>> fv; //vertex of the original mesh face
            std::vector<std::vector<float>> fv_p; //vertex of the parameter mesh face
            std::vector<bool> fv_flag; //whether the vertex is on the boundary
            std::vector<bool> fv_is_fixed; //whether the vertex is fixed
            std::vector<OpenMesh::VertexHandle> fvh; //save vertex handle in the triangle for border check

            //global phase equation: Ax = b;
            Eigen::Matrix2f J;//local fit
            //reset
            temp.setZero();
            B.setZero();
            X.setZero();

            typedef Eigen::Triplet<float> T;
            std::vector<T> triplet_list; //used to set A from temp

            for(auto f_it = halfedge_mesh->faces_begin(); f_it != halfedge_mesh->faces_end(); ++f_it)
            {
                test++;
                auto f_p = *f_it; //the corressponding face in para mesh
                //local phase
                //reset
                fv_p.clear();
                fv.clear();
                fv_idx.clear();
                fv_flag.clear();
                fv_is_fixed.clear();
                fvh.clear();
                J.setZero();
                //set fv& fv_flag
                for(auto fv_it = halfedge_mesh->fv_iter(*f_it); fv_it.is_valid(); ++fv_it)
                {
                    fvh.push_back(*fv_it);
                    std::vector<float> v2; //this vertex

                    for(int i = 0; i < 3; i++)
                    {
                        v2.push_back(halfedge_mesh->point(fv_it)[i]);
                    }
                    fv.push_back(v2);
                    fv_idx.push_back(fv_it->idx());
                }
                for(auto fv_it_p = parameter_mesh->fv_iter(f_p); fv_it_p.is_valid(); ++fv_it_p)
                {
                    std::vector<float> v1; //this vertex

                    for(int i = 0; i < 2; i++)
                    {
                        v1.push_back(parameter_mesh->point(fv_it_p)[i]);
                    }
                    fv_p.push_back(v1);

                    //set fv_is_fixed
                    if(fv_it_p->idx() == fixed_vertex)
                    {
                        fv_is_fixed.push_back(true);
                    } else
                    {
                        fv_is_fixed.push_back(false);
                    }
                    if(!set_reference)
                    {
                        std::vector<float> fixed_point;
                        for(int i = 0; i < 2; i++)
                            fixed_point.push_back(parameter_mesh->point(parameter_mesh->vertex_handle(fixed_vertex))[i]);
                        float distance = sqrt(pow(fixed_point[0] - v1[0],2) + pow(fixed_point[1] - v1[1],2));
                        if(distance > fixed_distance)
                        {
                            reference_vertex = fv_it_p->idx();
                            fixed_distance = distance;
                        }
                    }
                }
                for(int i = 0; i < 3; i++)
                {
                    int next_idx = (i + 1) % 3;
                    auto vh1 = fvh[i];
                    auto vh2 = fvh[next_idx];
                    auto he = halfedge_mesh->find_halfedge(vh1,vh2);
                    if(he.is_valid())
                    {
                        auto edge_h = halfedge_mesh->edge_handle(he);
                        if(halfedge_mesh->is_boundary(edge_h))
                        {
                            fv_flag.push_back(1);
                        } else
                        {
                            fv_flag.push_back(0);
                        }
                    }
                }
                if(fv_flag.size() != 3)
                    std::cout << "wrong edge flag" << std::endl;
                //----------------------boundary detection  checked----------------------------------//
                if(fv.size() != 3)
                {
                    throw std::runtime_error("Global phase: face vertex data is incomplete.");
                }
                std::shared_ptr<AXAP> local_fitter = std::make_shared<ARAP>(fv,fv_p); //solver for approx matrix
                J = local_fitter->localLinearFit(); //local R fit

                //----------------------local axis checked----------------------------------//
                //global phase
                //First, reform the original vertex-based equation in the paper to face based
                //solver for coefficient matrix
                if(n == 0)
                {
                    Eigen::Matrix3f A_local = local_fitter->coefficientsFromFace(fv,fv_flag,fv_is_fixed);
                    for(int i = 0; i < 3; i++)
                    {
                        for(int j = 0; j < 3; j++)
                        {

                            temp(fv_idx[i],fv_idx[j]) += A_local(i,j);

                        }
                    }
                }
                Eigen::MatrixXf B_local = local_fitter->bFromFace(J,fv,fv_p,fv_flag,fv_is_fixed);
                for(int i = 0; i < 3; i++)
                {
                    for(int j = 0; j < 2; j++)
                    {
                        B(fv_idx[i],j) += B_local(i,j);
                    }
                }
                if(test == 2 || test == 20 || test == 200)
                    std::cout << "local_B for test point: " << std::endl << B_local << std::endl;
            }
            for(int j = 0; j < 2; j++)
            {
                B(fixed_vertex,j) = parameter_mesh->point(parameter_mesh->vertex_handle(fixed_vertex))[j];

            }
            //fix the first vertex
            std::cout << "number of points:  " << n_points << std::endl;
            std::cout << "face iteration succeeded" << std::endl;

            //set A only in the first iteration
            if(n == 0)
            {
                for(int j = 0; j < n_points; j++)
                {
                    temp(fixed_vertex,j) = 0;

                }
                temp(fixed_vertex,fixed_vertex) = 1;

                for(int i = 0; i < n_points; i++)
                {
                    for(int j = 0; j < n_points; j++)
                    {
                        triplet_list.push_back(T(i,j,temp(i,j)));
                    }
                }
                A.setFromTriplets(triplet_list.begin(),triplet_list.end());
                solver.analyzePattern(A);
                solver.factorize(A);
                std::cout << "A factorized successfully" << std::endl;
            }

            X = solver.solve(B);
            std::cout << "X.rows() = " << X.rows() << ", X.cols() = " << X.cols() << std::endl;
            std::cout << "solution succeeded" << std::endl;
            std::cout << "reference vertex idx" << fixed_vertex << std::endl;
            Eigen::Vector2f v1(parameter_mesh->point(parameter_mesh->vertex_handle(reference_vertex))[0] -
                parameter_mesh->point(parameter_mesh->vertex_handle(fixed_vertex))[0],
                parameter_mesh->point(parameter_mesh->vertex_handle(reference_vertex))[1] -
                parameter_mesh->point(parameter_mesh->vertex_handle(fixed_vertex))[1]);

            Eigen::Vector2f v2(X(reference_vertex,0) -
                X(fixed_vertex,0),
                X(reference_vertex,1) -
                X(fixed_vertex,1));
            std::cout << "v12 set succeeded" << std::endl;
            Eigen::Vector2f v1_norm = v1.normalized();
            Eigen::Vector2f v2_norm = v2.normalized();


            float angle1 = std::atan2(v1_norm.y(),v1_norm.x());
            float angle2 = std::atan2(v2_norm.y(),v2_norm.x());
            float angle = angle2 - angle1;  // 
            std::cout << "angle:  " << angle<< std::endl;

            //R to fix the direction
            Eigen::Rotation2D<float> R(angle);
            std::cout << "R got" << std::endl;
            Eigen::Vector2f axis(parameter_mesh->point(parameter_mesh->vertex_handle(fixed_vertex))[0],
                parameter_mesh->point(parameter_mesh->vertex_handle(fixed_vertex))[1]);
            for(const auto& vertex : parameter_mesh->vertices())
            {
                int idx = vertex.idx();
                Eigen::Vector2f v(X(idx,0),X(idx,1));

                v = R.inverse() * (v - axis) + axis;
                parameter_mesh->set_point(vertex,{v(0),v(1),0});
            }
            std::cout << "iteration " << n << "succeeded" << std::endl;
            set_reference = true;
        }
    }

    // Set the output of the node
    auto geometry = openmesh_to_operand(parameter_mesh.get());

    // Set the output of the nodes
    params.set_output("Output",std::move(*geometry));
    return true;
}

NODE_DECLARATION_UI(arap_parameterization);
NODE_DEF_CLOSE_SCOPE

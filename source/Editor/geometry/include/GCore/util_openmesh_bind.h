#pragma once

#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <memory>

#include "GCore/GOP.h"

USTC_CG_NAMESPACE_OPEN_SCOPE
using PolyMesh = OpenMesh::PolyMesh_ArrayKernelT<>;
using TriMesh = OpenMesh::TriMesh_ArrayKernelT<>;

GEOMETRY_API std::shared_ptr<PolyMesh> operand_to_openmesh(
    Geometry* mesh_oeprand);

GEOMETRY_API std::shared_ptr<Geometry> openmesh_to_operand(PolyMesh* openmesh);

GEOMETRY_API std::shared_ptr<TriMesh> operand_to_openmesh_trimesh(
    Geometry* mesh_oeprand);

GEOMETRY_API std::shared_ptr<Geometry> openmesh_to_operand_trimesh(
    TriMesh* openmesh);

USTC_CG_NAMESPACE_CLOSE_SCOPE

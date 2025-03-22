#include <pxr/base/tf/stringUtils.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/basisCurves.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/points.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>

#include <string>

#include "GCore/Components/CurveComponent.h"
#include "GCore/Components/MaterialComponent.h"
#include "GCore/Components/MeshOperand.h"
#include "GCore/Components/PointsComponent.h"
#include "GCore/Components/XformComponent.h"
#include "GCore/geom_payload.hpp"
#include "geom_node_base.h"
#include "pxr/base/gf/rotation.h"
#include "pxr/usd/usd/payloads.h"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(write_usd)
{
    b.add_input<Geometry>("Geometry");
}

bool legal(const std::string& string)
{
    if (string.empty()) {
        return false;
    }
    if (std::find_if(string.begin(), string.end(), [](char val) {
            return val == '(' || val == ')' || val == '-' || val == ',';
        }) == string.end()) {
        return true;
    }
    return false;
}

NODE_EXECUTION_FUNCTION(write_usd)
{
    auto& global_payload = params.get_global_payload<GeomPayload&>();

    auto geometry = params.get_input<Geometry>("Geometry");

    auto mesh = geometry.get_component<MeshComponent>();

    auto points = geometry.get_component<PointsComponent>();

    auto curve = geometry.get_component<CurveComponent>();

    assert(!(points && mesh));

    pxr::UsdTimeCode time = global_payload.current_time;

    auto stage = global_payload.stage;
    auto sdf_path = global_payload.prim_path;

    stage->RemovePrim(sdf_path);

    if (mesh) {
        pxr::UsdGeomMesh usdgeom = pxr::UsdGeomMesh::Define(stage, sdf_path);
        if (usdgeom) {
#if USE_USD_SCRATCH_BUFFER
            copy_prim(mesh->get_usd_mesh().GetPrim(), usdgeom.GetPrim());
#else
            usdgeom.CreatePointsAttr().Set(mesh->get_vertices());
            usdgeom.CreateFaceVertexCountsAttr().Set(
                mesh->get_face_vertex_counts());
            usdgeom.CreateFaceVertexIndicesAttr().Set(
                mesh->get_face_vertex_indices());
            usdgeom.CreateNormalsAttr().Set(mesh->get_normals());
            if (!mesh->get_display_color().empty()) {
                auto primVarAPI = pxr::UsdGeomPrimvarsAPI(usdgeom);
                auto colorPrimvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken("displayColor"),
                    pxr::SdfValueTypeNames->Color3fArray);
                colorPrimvar.SetInterpolation(pxr::UsdGeomTokens->vertex);
                colorPrimvar.Set(mesh->get_display_color());
            }
            if (!mesh->get_texcoords_array().empty()) {
                auto primVarAPI = pxr::UsdGeomPrimvarsAPI(usdgeom);
                auto primvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken("UVMap"),
                    pxr::SdfValueTypeNames->TexCoord2fArray);
                primvar.Set(mesh->get_texcoords_array());
                if (mesh->get_texcoords_array().size() ==
                    mesh->get_vertices().size()) {
                    primvar.SetInterpolation(pxr::UsdGeomTokens->vertex);
                }
                else {
                    primvar.SetInterpolation(pxr::UsdGeomTokens->faceVarying);
                }
            }

#endif
            usdgeom.CreateDoubleSidedAttr().Set(true);

            // Store polyscope quantities
            auto primVarAPI = pxr::UsdGeomPrimvarsAPI(usdgeom);

            // It's invalid to use pxr::TfToken("some_prefix" + some_string)
            // directly, so we need to create a new string first.

            for (const std::string& name :
                 mesh->get_vertex_scalar_quantity_names()) {
                auto values = mesh->get_vertex_scalar_quantity(name);
                const std::string primvar_name =
                    "polyscope:vertex:scalar:" + name;
                auto primvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken(primvar_name.c_str()),
                    pxr::SdfValueTypeNames->FloatArray);
                primvar.SetInterpolation(pxr::UsdGeomTokens->vertex);
                primvar.Set(values);
            }

            for (const std::string& name :
                 mesh->get_face_scalar_quantity_names()) {
                auto values = mesh->get_face_scalar_quantity(name);
                const std::string primvar_name =
                    "polyscope:face:scalar:" + name;
                auto primvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken(primvar_name.c_str()),
                    pxr::SdfValueTypeNames->FloatArray);
                primvar.SetInterpolation(pxr::UsdGeomTokens->uniform);
                primvar.Set(values);
            }

            for (std::string& name : mesh->get_vertex_color_quantity_names()) {
                auto values = mesh->get_vertex_color_quantity(name);
                const std::string primvar_name =
                    "polyscope:vertex:color:" + name;
                auto primvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken(primvar_name.c_str()),
                    pxr::SdfValueTypeNames->Color3fArray);
                primvar.SetInterpolation(pxr::UsdGeomTokens->vertex);
                primvar.Set(values);
            }

            for (const std::string& name :
                 mesh->get_face_color_quantity_names()) {
                auto values = mesh->get_face_color_quantity(name);
                const std::string primvar_name = "polyscope:face:color:" + name;
                auto primvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken(primvar_name.c_str()),
                    pxr::SdfValueTypeNames->Color3fArray);
                primvar.SetInterpolation(pxr::UsdGeomTokens->uniform);
                primvar.Set(values);
            }

            for (const std::string& name :
                 mesh->get_vertex_vector_quantity_names()) {
                auto values = mesh->get_vertex_vector_quantity(name);
                const std::string primvar_name =
                    "polyscope:vertex:vector:" + name;
                auto primvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken(primvar_name.c_str()),
                    pxr::SdfValueTypeNames->Vector3fArray);
                primvar.SetInterpolation(pxr::UsdGeomTokens->vertex);
                primvar.Set(values);
            }

            for (const std::string& name :
                 mesh->get_face_vector_quantity_names()) {
                auto values = mesh->get_face_vector_quantity(name);
                const std::string primvar_name =
                    "polyscope:face:vector:" + name;
                auto primvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken(primvar_name.c_str()),
                    pxr::SdfValueTypeNames->Vector3fArray);
                primvar.SetInterpolation(pxr::UsdGeomTokens->uniform);
                primvar.Set(values);
            }

            for (const std::string& name :
                 mesh->get_face_corner_parameterization_quantity_names()) {
                auto values =
                    mesh->get_face_corner_parameterization_quantity(name);
                const std::string primvar_name =
                    "polyscope:face_corner:parameterization:" + name;
                auto primvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken(primvar_name.c_str()),
                    pxr::SdfValueTypeNames->TexCoord2fArray);
                primvar.SetInterpolation(pxr::UsdGeomTokens->faceVarying);
                primvar.Set(values);
            }

            for (const std::string& name :
                 mesh->get_vertex_parameterization_quantity_names()) {
                auto values = mesh->get_vertex_parameterization_quantity(name);
                const std::string primvar_name =
                    "polyscope:vertex:parameterization:" + name;
                auto primvar = primVarAPI.CreatePrimvar(
                    pxr::TfToken(primvar_name.c_str()),
                    pxr::SdfValueTypeNames->TexCoord2fArray);
                primvar.SetInterpolation(pxr::UsdGeomTokens->vertex);
                primvar.Set(values);
            }
        }
    }
    else if (points) {
        pxr::UsdGeomPoints usdpoints =
            pxr::UsdGeomPoints::Define(stage, sdf_path);

        usdpoints.CreatePointsAttr().Set(points->get_vertices(), time);

        if (points->get_width().size() > 0) {
            usdpoints.CreateWidthsAttr().Set(points->get_width(), time);
        }

        auto PrimVarAPI = pxr::UsdGeomPrimvarsAPI(usdpoints);
        if (points->get_display_color().size() > 0) {
            pxr::UsdGeomPrimvar colorPrimvar = PrimVarAPI.CreatePrimvar(
                pxr::TfToken("displayColor"),
                pxr::SdfValueTypeNames->Color3fArray);
            colorPrimvar.SetInterpolation(pxr::UsdGeomTokens->vertex);
            colorPrimvar.Set(points->get_display_color(), time);
        }
    }
    else if (curve) {
        pxr::UsdGeomBasisCurves usd_curve =
            pxr::UsdGeomBasisCurves::Define(stage, sdf_path);
        if (usd_curve) {
#if USE_USD_SCRATCH_BUFFER
            copy_prim(curve->get_usd_curve().GetPrim(), usd_curve.GetPrim());
#else
            usd_curve.CreatePointsAttr().Set(curve->get_vertices());
            usd_curve.CreateWidthsAttr().Set(curve->get_width());
            usd_curve.CreateCurveVertexCountsAttr().Set(
                curve->get_vert_count());
            usd_curve.CreateNormalsAttr().Set(curve->get_curve_normals());
            usd_curve.CreateDisplayColorAttr().Set(curve->get_display_color());
            usd_curve.CreateWrapAttr().Set(
                curve->get_periodic() ? pxr::UsdGeomTokens->periodic
                                      : pxr::UsdGeomTokens->nonperiodic);
#endif
        }
    }

    // Material and Texture
    auto material_component = geometry.get_component<MaterialComponent>();
    if (material_component) {
        auto usdgeom = pxr::UsdGeomXformable ::Get(stage, sdf_path);
        if (legal(std::string(material_component->textures[0].c_str()))) {
            auto texture_name =
                std::string(material_component->textures[0].c_str());
            std::filesystem::path p =
                std::filesystem::path(texture_name).replace_extension();
            auto file_name = "texture" + p.filename().string();

            auto material_path_root = pxr::SdfPath("/TexModel");
            auto material_path =
                material_path_root.AppendPath(pxr::SdfPath(file_name + "Mat"));
            auto material_shader_path =
                material_path.AppendPath(pxr::SdfPath("PBRShader"));
            auto material_stReader_path =
                material_path.AppendPath(pxr::SdfPath("stReader"));
            auto material_texture_path =
                material_path.AppendPath(pxr::SdfPath("diffuseTexture"));

            auto material = pxr::UsdShadeMaterial::Define(stage, material_path);
            auto pbrShader =
                pxr::UsdShadeShader::Define(stage, material_shader_path);

            pbrShader.CreateIdAttr(
                pxr::VtValue(pxr::TfToken("UsdPreviewSurface")));
            material.CreateSurfaceOutput().ConnectToSource(
                pbrShader.ConnectableAPI(), pxr::TfToken("surface"));

            auto stReader =
                pxr::UsdShadeShader::Define(stage, material_stReader_path);
            stReader.CreateIdAttr(
                pxr::VtValue(pxr::TfToken("UsdPrimvarReader_float2")));

            auto diffuseTextureSampler =
                pxr::UsdShadeShader::Define(stage, material_texture_path);

            diffuseTextureSampler.CreateIdAttr(
                pxr::VtValue(pxr::TfToken("UsdUVTexture")));
            diffuseTextureSampler
                .CreateInput(
                    pxr::TfToken("file"), pxr::SdfValueTypeNames->Asset)
                .Set(pxr::SdfAssetPath(texture_name));
            diffuseTextureSampler
                .CreateInput(pxr::TfToken("st"), pxr::SdfValueTypeNames->Float2)
                .ConnectToSource(
                    stReader.ConnectableAPI(), pxr::TfToken("result"));
            diffuseTextureSampler.CreateOutput(
                pxr::TfToken("rgb"), pxr::SdfValueTypeNames->Float3);

            diffuseTextureSampler
                .CreateInput(
                    pxr::TfToken("wrapS"), pxr::SdfValueTypeNames->Token)
                .Set(pxr::TfToken("mirror"));

            diffuseTextureSampler
                .CreateInput(
                    pxr::TfToken("wrapT"), pxr::SdfValueTypeNames->Token)
                .Set(pxr::TfToken("mirror"));

            pbrShader
                .CreateInput(
                    pxr::TfToken("diffuseColor"),
                    pxr::SdfValueTypeNames->Color3f)
                .ConnectToSource(
                    diffuseTextureSampler.ConnectableAPI(),
                    pxr::TfToken("rgb"));

            auto stInput = material.CreateInput(
                pxr::TfToken("frame:stPrimvarName"),
                pxr::SdfValueTypeNames->Token);
            stInput.Set(pxr::TfToken("UVMap"));

            stReader
                .CreateInput(
                    pxr::TfToken("varname"), pxr::SdfValueTypeNames->Token)
                .ConnectToSource(stInput);

            usdgeom.GetPrim().ApplyAPI(pxr::UsdShadeTokens->MaterialBindingAPI);
            pxr::UsdShadeMaterialBindingAPI(usdgeom).Bind(material);
        }
        else {
            // TODO: Throw something
        }
    }

    auto xform_component = geometry.get_component<XformComponent>();
    if (xform_component) {
        auto usdgeom = pxr::UsdGeomXformable ::Get(stage, sdf_path);
        // Transform
        assert(
            xform_component->translation.size() ==
            xform_component->rotation.size());

        pxr::GfMatrix4d final_transform = xform_component->get_transform();

        auto xform_op = usdgeom.GetTransformOp();
        if (!xform_op) {
            xform_op = usdgeom.AddTransformOp();
        }
        xform_op.Set(final_transform, time);
    }
    else {
        auto usdgeom = pxr::UsdGeomXformable ::Get(stage, sdf_path);
        auto xform_op = usdgeom.GetTransformOp();
        if (!xform_op) {
            xform_op = usdgeom.AddTransformOp();
        }
        xform_op.Set(pxr::GfMatrix4d(1), time);
    }

    if (global_payload.has_simulation) {
        pxr::UsdPrim prim = stage->GetPrimAtPath(sdf_path);
        prim.CreateAttribute(
                pxr::TfToken("Animatable"), pxr::SdfValueTypeNames->Bool)
            .Set(true);
    }
    else {
        pxr::UsdPrim prim = stage->GetPrimAtPath(sdf_path);
        prim.CreateAttribute(
                pxr::TfToken("Animatable"), pxr::SdfValueTypeNames->Bool)
            .Set(false);
    }

    pxr::UsdGeomImageable(stage->GetPrimAtPath(sdf_path)).MakeVisible();
    return true;
}

NODE_DECLARATION_REQUIRED(write_usd);

NODE_DECLARATION_UI(write_usd);
NODE_DEF_CLOSE_SCOPE

//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_SLANGSHADERGENERATOR_H
#define MATERIALX_SLANGSHADERGENERATOR_H

/// @file
/// SLANG shader generator

#include <MaterialXGenShader/HwShaderGenerator.h>

#include "Export.h"

MATERIALX_NAMESPACE_BEGIN

using SlangShaderGeneratorPtr = shared_ptr<class SlangShaderGenerator>;

/// Base class for SLANG (OpenGL Shading Language) code generation.
/// A generator for a specific SLANG target should be derived from this class.
class HD_USTC_CG_API SlangShaderGenerator : public HwShaderGenerator {
   public:
    SlangShaderGenerator();

    static ShaderGeneratorPtr create()
    {
        return std::make_shared<SlangShaderGenerator>();
    }

    /// Generate a shader starting from the given element, translating
    /// the element and all dependencies upstream into shader code.
    ShaderPtr generate(
        const string& name,
        ElementPtr element,
        GenContext& context) const override;

    /// Return a unique identifier for the target this generator is for
    const string& getTarget() const override
    {
        return TARGET;
    }

    /// Return the version string for the SLANG version this generator is for
    virtual const string& getVersion() const
    {
        return VERSION;
    }

    /// Emit a shader variable.
    void emitVariableDeclaration(
        const ShaderPort* variable,
        const string& qualifier,
        GenContext& context,
        ShaderStage& stage,
        bool assignValue = true) const override;

    /// Return a registered shader node implementation given an implementation
    /// element. The element must be an Implementation or a NodeGraph acting as
    /// implementation.
    ShaderNodeImplPtr getImplementation(
        const NodeDef& nodedef,
        GenContext& context) const override;

    // For slang, use import instead
    void emitLibraryInclude(
        const FilePath& filename,
        GenContext& context,
        ShaderStage& stage) const override;

    void emitBlock(
        const string& str,
        const FilePath& sourceFilename,
        GenContext& context,
        ShaderStage& stage) const override;

    /// Determine the prefix of vertex data variables.
    string getVertexDataPrefix(const VariableBlock& vertexData) const override;

   public:
    /// Unique identifier for this generator target
    static const string TARGET;

    /// Version string for the generator target
    static const string VERSION;

   protected:
    virtual void emitVertexStage(
        const ShaderGraph& graph,
        GenContext& context,
        ShaderStage& stage) const;
    virtual void emitPixelStage(
        const ShaderGraph& graph,
        GenContext& context,
        ShaderStage& stage) const;

    virtual void emitDirectives(GenContext& context, ShaderStage& stage) const;
    virtual void emitConstants(GenContext& context, ShaderStage& stage) const;
    virtual void emitUniforms(GenContext& context, ShaderStage& stage) const;
    virtual void emitLightData(GenContext& context, ShaderStage& stage) const;
    virtual void emitInputs(GenContext& context, ShaderStage& stage) const;
    virtual void emitOutputs(GenContext& context, ShaderStage& stage) const;

    virtual HwResourceBindingContextPtr getResourceBindingContext(
        GenContext& context) const;

    /// Logic to indicate whether code to support direct lighting should be
    /// emitted. By default if the graph is classified as a shader, or BSDF node
    /// then lighting is assumed to be required. Derived classes can override
    /// this logic.
    virtual bool requiresLighting(const ShaderGraph& graph) const;

    /// Emit specular environment lookup code
    virtual void emitSpecularEnvironment(
        GenContext& context,
        ShaderStage& stage) const;

    /// Emit transmission rendering code
    virtual void emitTransmissionRender(GenContext& context, ShaderStage& stage)
        const;

    /// Emit function definitions for lighting code
    virtual void emitLightFunctionDefinitions(
        const ShaderGraph& graph,
        GenContext& context,
        ShaderStage& stage) const;

    static void toVec4(const TypeDesc* type, string& variable);

    /// Nodes used internally for light sampling.
    vector<ShaderNodePtr> _lightSamplingNodes;
};

/// Base class for common SLANG node implementations
class HD_USTC_CG_API SlangImplementation : public HwImplementation {
   public:
    const string& getTarget() const override;

   protected:
    SlangImplementation()
    {
    }
};

MATERIALX_NAMESPACE_END

#endif

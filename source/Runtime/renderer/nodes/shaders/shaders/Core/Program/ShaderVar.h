/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once
#include <cstddef>
#include <memory>
#include <string_view>

#include "Core/Macros.h"
#include "ProgramReflection.h"
#include "utils/Math/Vector.h"

namespace USTC_CG {
class ParameterBlock;

/**
 * A "pointer" to a shader variable stored in some parameter block.
 *
 * A `ShaderVar` works like a pointer to the data "inside" a `ParameterBlock`.
 * It keeps track of three things:
 *
 * 1. The parameter block that is being pointed into
 * 2. An offset into the data of that parameter block
 * 3. The type of the data at that offset
 *
 * Typically a `ShaderVar` is created using the `getRootVar()` operation
 * on `ParameterBlock`, which yields a shader variable that points to
 * the entire "contents" of the parameter block.
 *
 * Given a `ShaderVar` that represents a value with `struct` or
 * array type, we can use `operator[]` to get a shader variable
 * that points to a single `struct` field or array element:
 *
 * // Shader code has `MyStruct myVar[10];`
 *
 * ShaderVar myVar = pObj["myVar"];                // works like &myVar
 * ShaderVar arrayElement = myVar[2];              // works like &myVar[2]
 * ShaderVar someField = arrayElement["someField"] // works like
 * &myVar[2].someField
 *
 * Once you have a `ShaderVar` that refers to a simple value you
 * want to set, you can do so with either an explicit `set*()` function
 * or an overload of `operator=`:
 *
 * someField = float3(0);
 *
 * pObj["someTexture"].setTexture(pMyTexture);
 */
struct HD_USTC_CG_API ShaderVar {
    /**
     * Create a null/invalid shader variable pointer.
     */
    ShaderVar();

    /**
     * Copy constructor.
     */
    ShaderVar(const ShaderVar& other);

    /**
     * Create a shader variable pointer into `pObject` at the given `offset`.
     */
    explicit ShaderVar(
        ParameterBlock* pObject,
        const TypedShaderVarOffset& offset);

    /**
     * Create a shader variable pointer to the content of `pObject`.
     */
    explicit ShaderVar(ParameterBlock* pObject);

    /**
     * Check if this shader variable pointer is valid/non-null.
     */
    bool isValid() const
    {
        return mOffset.isValid();
    }

    /**
     * Get the type data this shader variable points at.
     *
     * For an invalid/null shader variable the result will be null.
     */
    const ReflectionType* getType() const
    {
        return mOffset.getType();
    }

    /**
     * Get the offset that this shader variable points to inside the parameter
     * block.
     */
    TypedShaderVarOffset getOffset() const
    {
        return mOffset;
    }

    /**
     * Get the byte offset that this shader variable points to inside the
     * parameter block.
     *
     * Note: If the type of the value being pointed at includes anything other
     * than ordinary/uniform data, then this byte offset will not provide
     * complete enough information to re-create the same `ShaderVar` later.
     */
    size_t getByteOffset() const
    {
        return mOffset.getUniform().getByteOffset();
    }

    //
    // Navigation
    //

    /**
     * Get a shader variable pointer to a sub-field.
     *
     * This shader variable must point to a value of `struct` type,
     * with a field matching the given `name`.
     *
     * If this shader variable points at a constant buffer or parameter block,
     * then the lookup will proceed in the contents of that block.
     *
     * If the above doesn't hold, an exception is thrown.
     */
    ShaderVar operator[](std::string_view name) const;

    /**
     * Get a shader variable pointer to an element or sub-field.
     *
     * This operation is valid in two cases:
     * 1) This shader variable points at a value of array type, and the `index`
     * is in range for the array.
     * The result is a shader variable that points to a single array element.
     * 2) This shader variable points at a value of `struct` type, and `index`
     * is in range for the number of fields in the `struct`.
     * The result is a shader variable that points to a single `struct` field.
     * If this shader variable points at a constant buffer or parameter block,
     * then the lookup will proceed in the contents of that block.
     *
     * If the above doesn't hold, an exception is thrown.
     */
    ShaderVar operator[](size_t index) const;

    /**
     * Try to get a variable for a member/field.
     *
     * Unlike `operator[]`, a `findMember` operation does not throw an exception
     * if the variable doesn't exist. Instead, it returns an invalid
     * `ShaderVar`.
     */
    ShaderVar findMember(std::string_view name) const;

    /**
     * Returns true if a member/field exists.
     */
    bool hasMember(std::string_view name) const
    {
        return findMember(name).isValid();
    }

    /**
     * Try to get a variable for a member/field, by index.
     *
     * Unlike `operator[]`, a `findMember` operation does not throw an exception
     * if the varible doesn't exist. Instead, it returns an invalid `ShaderVar`.
     */
    ShaderVar findMember(uint32_t index) const;

    /**
     * Returns true if a member/field exists, by index.
     */
    bool hasMember(uint32_t index) const
    {
        return findMember(index).isValid();
    }

    //
    // Variable assignment
    //

    /**
     * Set the value of the data pointed to by this shader variable.
     * Throws an exception if the given `val` does not have a suitable type for
     * the value pointed to by this shader variable.
     */
    template<typename T>
    void set(const T& val) const
    {
        setImpl<T>(val);
    }

    /**
     * Set the value of the data pointed to by this shader variable.
     *
     * This operator allows assignment syntax to be used in place of
     * the `set()` method. The following two statements are equivalent:
     * myShaderVar["someField"].set(float4(0));
     * myShaderVar["someField"] = float4(0);
     *
     * Throws an exception if the given `val` does not have a suitable type for
     * the value pointed to by this shader variable.
     */
    template<typename T>
    void operator=(const T& val) const
    {
        setImpl(val);
    }

    //
    // Uniforms
    //

    /**
     * Assign raw binary data to the variable.
     *
     * This operation will only assign to the ordinary/"uniform" data pointed
     * to by this shader variable, and will not affect any nested variables
     * of texture/buffer/sampler types.
     *
     * Throws an exception if this variable doesn't point at a parameter block
     * or constant buffer.
     */
    void setBlob(void const* data, size_t size) const;

    /**
     * Assign raw binary data to the variable.
     * This is a convenience form for `setBlob(&val, sizeof(val)`.
     */
    template<typename T>
    void setBlob(const T& val) const
    {
        setBlob(&val, sizeof(val));
    }

    //
    // Resource binding
    //

    /**
     * Bind a buffer to this variable.
     * Throws an exception if this variable doesn't point at a buffer or the
     * buffer has incompatible bind flags.
     */
    void setBuffer(const nvrhi::BufferHandle& pBuffer) const;

    /**
     * Get the buffer bound to this variable.
     * Throws an exception if this variable doesn't point at a buffer.
     */
    nvrhi::BufferHandle getBuffer() const;

    /**
     * Implicit conversion from a shader variable to a buffer.
     * This operation allows a bound buffer to be queried using the `[]` syntax:
     * pBuffer = pVars["someBuffer"];
     */
    operator nvrhi::BufferHandle() const
    {
        return getBuffer();
    }

    /**
     * Bind a texture to this variable.
     * Throws an exception if this variable doesn't point at a texture or the
     * texture has incompatible bind flags.
     */
    void setTexture(const nvrhi::TextureHandle& pTexture) const;

    /**
     * Get the texture bound to this variable.
     * Throws an exception if this variable doesn't point at a texture.
     */
    nvrhi::TextureHandle getTexture() const;

    /**
     * Implicit conversion from a shader variable to a texture.
     * This operation allows a bound texture to be queried using the `[]`
     * syntax: pTexture = pVars["someTexture"];
     */
    operator nvrhi::TextureHandle() const
    {
        return getTexture();
    }

    /**
     * Bind an SRV to this variable.
     * Throws an exception if this variable doesn't point at an SRV.
     */
    void setSrv(const nvrhi::BindingSetItem& pSrv) const;

    /**
     * Get the SRV bound to this variable.
     * Throws an exception if this variable doesn't point at an SRV.
     */
    nvrhi::BindingSetItem getSrv() const;

    /**
     * Bind a UAV to this variable.
     * Throws an exception if this variable doesn't point at a UAV.
     */
    void setUav(const nvrhi::BindingSetItem& pUav) const;

    /**
     * Get the UAV bound to this variable.
     * Throws an exception if this variable doesn't point at a UAV.
     */
    nvrhi::BindingSetItem getUav() const;

    /**
     * Bind an acceleration structure to this variable.
     * Throws an exception if this variable doesn't point at an acceleration
     * structure.
     */
    void setAccelerationStructure(
        const nvrhi::rt::AccelStructHandle& pAccl) const;

    /**
     * Get the acceleration structure bound to this variable.
     * Throws an exception if this variable doesn't point at an acceleration
     * structure.
     */
    nvrhi::rt::AccelStructHandle getAccelerationStructure() const;

    /**
     * Bind a sampler to this variable.
     * Throws an exception if this variable doesn't point at a sampler.
     */
    void setSampler(const nvrhi::SamplerHandle& pSampler) const;

    /**
     * Get the sampler bound to this variable.
     * Throws an exception if this variable doesn't point at a sampler.
     */
    nvrhi::SamplerHandle getSampler() const;

    /**
     * Implicit conversion from a shader variable to a sampler.
     * This operation allows a bound sampler to be queried using the `[]`
     * syntax: pSampler = pVars["someSampler"];
     */
    operator nvrhi::SamplerHandle() const;

    /**
     * Bind a parameter block to this variable.
     * Throws an exception if this variable doesn't point at a parameter block.
     */
    void setParameterBlock(const ref<ParameterBlock>& pBlock) const;

    /**
     * Get the parameter block bound to this variable.
     * Throws an exception if this variable doesn't point at a parameter block.
     */
    ref<ParameterBlock> getParameterBlock() const;

    //
    // Offset access
    //

    /**
     * Implicit conversion from a shader variable to its offset information.
     *
     * This operation allows the offset information for a shader variable
     * to be queried easily using the `[]` sugar:
     *
     * TypedShaderVarOffset myVarLoc = pVars["myVar"];
     * ...
     * pVars[myVarLoc] = someValue
     *
     * Note that the returned offset information only retains the offset into
     * the leaf-most parameter block (constant buffer or parameter block).
     * Users must take care when using an offset that they apply the offset
     * to the correct object:
     *
     * auto pPerFrameCB = pVars["PerFrameCB"];
     * TypedShaderVarOffset myVarLoc = pPerFrameCB["myVar"];
     * ...
     * pVars[myVarLoc] = someValue; // CRASH!
     */
    operator TypedShaderVarOffset() const
    {
        return mOffset;
    }

    /**
     * Implicit conversion from a shader variable to its offset information.
     *
     * This operation allows the offset information for a shader variable
     * to be queried easily using the `[]` sugar:
     *
     * UniformShaderVarOffset myVarLoc = pVars["myVar"];
     * ...
     * pVars[myVarLoc] = someValue
     *
     * Note that the returned offset information only retains the offset into
     * the leaf-most parameter block (constant buffer or parameter block).
     * Users must take care when using an offset that they apply the offset
     * to the correct object:
     *
     * auto pPerFrameCB = pVars["PerFrameCB"];
     * UniformShaderVarOffset myVarLoc = pPerFrameCB["myVar"];
     * ...
     * pVars[myVarLoc] = someValue; // CRASH!
     */
    operator UniformShaderVarOffset() const
    {
        return mOffset.getUniform();
    }

    /**
     * Create a shader variable that points to some pre-computed `offset`
     * relative to this one.
     *
     * This operation assumes that the provided `offset` has been appropriately
     * computed based on a type that matches what this shader variable points
     * to.
     *
     * The resulting shader variable will have the type encoded in `offset`,
     * and will have an offset that is the sum of this variables offset
     * and the provided `offset`.
     */
    ShaderVar operator[](const TypedShaderVarOffset& offset) const;

    /**
     * Create a shader variable that points to some pre-computed `offset`
     * relative to this one.
     *
     * This operation assumes that the provided `offset` has been appropriately
     * computed based on a type that matches what this shader variable points
     * to.
     *
     * Because a `UniformShaderVarOffset` does not encode type information,
     * this operation will search for a field/element matching the given
     * `offset` and use its type information in the resulting shader variable.
     * If no appropriate field/element can be found, an error will be logged.
     */
    ShaderVar operator[](const UniformShaderVarOffset& offset) const;

    /**
     * Get access to the underlying bytes of the variable.
     *
     * This operation must be used with caution,
     * the caller takes all responsibility for validation.
     *
     * Note: if a caller uses the resulting pointer to write to the variable
     * (e.g. by casting away the `const`-ness, then the underlying
     * `ParameterBlock` will not automatically be marked dirty, and it is
     * possible that the effects of that write will not be visible.
     */
    void const* getRawData() const;

   private:
    /**
     * The parameter block that is being pointed into.
     *
     * Note: this is an unowned pointer, so it is *not* safe to hold onto
     * a `ShaderVar` for long periods of time where the object it points into
     * might get released. This is a concession to performance, since we do not
     * want to perform reference-counting each and every time a `ShaderVar`
     * gets created or destroyed.
     */
    ParameterBlock* mpBlock;

    /**
     * The offset into the object where this variable points.
     *
     * This field encodes both the offset information and the type of the
     * variable.
     */
    TypedShaderVarOffset mOffset;

    void setImpl(const nvrhi::TextureHandle& pTexture) const;
    void setImpl(const nvrhi::SamplerHandle& pSampler) const;
    void setImpl(const nvrhi::BufferHandle& pBuffer) const;
    void setImpl(const ref<ParameterBlock>& pBlock) const;

    template<typename T>
    void setImpl(const T& val) const;
};
}  // namespace USTC_CG

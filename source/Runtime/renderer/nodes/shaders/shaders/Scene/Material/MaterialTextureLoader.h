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
#include "Core/Macros.h"
#include "Scene/Material/Material.h"
#include "utils/Image/TextureManager.h"
#include <filesystem>
#include <vector>

namespace USTC_CG
{
    /** Helper class to load material textures using the texture manager.

        Calling `loadTexture` does not assign the texture to the material right away.
        Instead, an asynchronous texture load request is issued and a reference for the
        material assignment is stored. When the client destroys the instance of the
        `MaterialTextureLoader`, it blocks until all textures are loaded and assigns
        them to the materials.
    */
    class HD_USTC_CG_API MaterialTextureLoader
    {
    public:
        MaterialTextureLoader(TextureManager& textureManager, bool useSrgb);
        ~MaterialTextureLoader();

        /** Request loading a material texture.
            \param[in] pMaterial Material to load texture into.
            \param[in] slot Slot to load texture into.
            \param[in] path Texture file path.
        */
        void loadTexture(const ref<Material>& pMaterial, Material::TextureSlot slot, const std::filesystem::path& path);

        void finishLoading()
        {
            assignTextures();
        }
    private:
        void assignTextures();

        struct TextureAssignment
        {
            ref<Material> pMaterial;
            Material::TextureSlot textureSlot;
            TextureManager::CpuTextureHandle handle;
        };

        bool mUseSrgb;
        std::vector<TextureAssignment> mTextureAssignments;
        TextureManager& mTextureManager;
    };
}

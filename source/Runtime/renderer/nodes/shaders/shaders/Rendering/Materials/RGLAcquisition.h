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

#include "Core/Pass/ComputePass.h"
#include "Scene/Scene.h"
#include "Scene/Material/RGLFile.h"
#include <memory>

namespace USTC_CG
{
    /** This class allows taking a virtual measurement of a BRDF
        and converting it into the parametrization proposed by
        Dupuy & Jakob,  "An Adaptive Parameterization for Efficient
        Material Acquisition and Rendering".
    */
    class HD_USTC_CG_API RGLAcquisition
    {
    public:
        /// Constructor.
        RGLAcquisition(ref<Device> pDevice, const ref<Scene>& pScene);

        void acquireIsotropic(RenderContext* pRenderContext, const MaterialID materialID);
        RGLFile toRGLFile();

    private:
        ref<Device> mpDevice;
        ref<Scene> mpScene;
        ref<ComputePass> mpRetroReflectionPass;
        ref<ComputePass> mpBuildKernelPass;
        ref<ComputePass> mpPowerIterationPass;
        ref<ComputePass> mpIntegrateSigmaPass;
        ref<ComputePass> mpSumSigmaPass;
        ref<ComputePass> mpComputeThetaPass;
        ref<ComputePass> mpComputeVNDFPass;
        ref<ComputePass> mpAcquireBRDFPass;

        nvrhi::BufferHandle mpNDFDirectionsBuffer;      ///< Stores hemispherical directions of entries in the NDF table.
        nvrhi::BufferHandle mpRetroBuffer;              ///< 2D table storing measured retroreflecton of BRDF.
        nvrhi::BufferHandle mpNDFKernelBuffer;          ///< Stores kernel matrix of Fredholm problem for retrieving NDF.
        nvrhi::BufferHandle mpNDFBuffer;                ///< 2D table storing the retrieved NDF.
        nvrhi::BufferHandle mpNDFBufferTmp;
        nvrhi::BufferHandle mpSigmaBuffer;              ///< 2D table of projected microfacet area, integrated numerically.
        nvrhi::BufferHandle mpThetaBuffer;              ///< 1D tables storing angles at which measurements are taken.
        nvrhi::BufferHandle mpPhiBuffer;
        nvrhi::BufferHandle mpVNDFBuffer;               ///< 4D table (over wi x wo domains) containing the visible distribution of normals.
        nvrhi::BufferHandle mpVNDFMargBuffer;           ///< Auxiliary buffers for sampling the VNDF.
        nvrhi::BufferHandle mpVNDFCondBuffer;
        nvrhi::BufferHandle mpLumiBuffer;               ///< 4D table of measured luminance and RGB reflectance of RGB.
        nvrhi::BufferHandle mpRGBBuffer;
    };
}

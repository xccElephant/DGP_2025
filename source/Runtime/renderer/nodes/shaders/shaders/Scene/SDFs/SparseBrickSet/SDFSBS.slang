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
#include "utils/Math/MathConstants.slangh"

import Scene.SDFs.SDFGridBase;
import Scene.SDFs.SDFVoxelCommon;
import Scene.SDFs.SDFVoxelHitUtils;
import Utils.Math.AABB;
import Utils.Math.FormatConversion;
import Utils.Math.PackedFormats;
import Utils.Geometry.IntersectionHelpers;

#define USE_GATHER 1

struct SDFSBS : SDFGridBase, ISDFGrid
{

    StructuredBuffer<AABB> aabbs;
    Texture3D<uint> indirectionBuffer;
    Texture2D<float> bricks;
    SamplerState sampler;

    uint virtualGridWidth;
    uint virtualBricksPerAxis;
    uint2 bricksPerAxis;
    uint2 brickTextureDimensions;
    uint brickWidth;
    float normalizationFactor;

    bool calculateBrickCoords(float3 pLocal, float3 brickMinLocal, float3 brickMaxLocal, out uint3 virtualBrickCoords, out uint2 brickTexelCoords, out float3 virtualVoxelBrickPos, out float3 voxelUnitCoords, out int3 brickLocalMinCoords, out int3 brickLocalMaxCoords)
    {
        // Important note, the grid in "brick space" might be larger than the grid in "voxel space":
        // |-----------------------|--|
        // |-----------------------|--| <--- These additional slivers on the top and right are in brick space [0, virtualBricksPerAxis * brickWidth)
        // |                       |  |      Brick space is the same size as voxel space if virtualBricksPerAxis * brickWidth = virtualGridWidth.
        // |                       |  |
        // |                       |  |
        // |      Voxel Space      |  |
        // | [0, virtualGridWidth) |  |
        // |                       |  |
        // |                       |  |
        // |-----------------------|--|
        // 0      Local Space      1

        // We have the AABB of the brick that we hit, use the center of that
        // to calculate brick coordinates.
        float3 brickCenterLocal = (brickMinLocal + brickMaxLocal) * 0.5f;

        // Calculate the bricks center position in the virtual brick grid: [0, virtualBricksPerAxis),
        // we can't multiply by virtualBricksPerAxis as brickCenterLocal is in the local space of the SDF
        // which may be smaller than the brick space, see comment above.
        float3 brickCenterInVirtualBrickGrid = brickCenterLocal * float(virtualGridWidth) / brickWidth;

        // Floor to get virtual brick coords.
        virtualBrickCoords = uint3(brickCenterInVirtualBrickGrid);
        int3 brickCoordsInVirtualVoxelGrid = virtualBrickCoords * brickWidth;

        // Calculate the position that in the virtual voxel grid.
        float3 hitPosInVirtualVoxelGrid = pLocal * virtualGridWidth;

        // Precision issues may cause hitPosInVirtualVoxelGrid to appear in the wrong brick.
        // We need to clamp the voxel to be within the brick we hit.
        // We need to take into account that bricks may not be the full brickWidth in size.
        // We can convert the brick AABB to voxel dimensions and "round up".
        int3 brickMinCoordsInVirtualVoxelGrid = int3(floor(brickMinLocal * virtualGridWidth));
        int3 brickMaxCoordsInVirtualVoxelGrid = int3(ceil(brickMaxLocal * virtualGridWidth) - 1);

        // When DDAing through the brick, never allow it to access voxels outside these bounds.
        brickLocalMinCoords = brickMinCoordsInVirtualVoxelGrid - brickCoordsInVirtualVoxelGrid;
        brickLocalMaxCoords = brickMaxCoordsInVirtualVoxelGrid - brickCoordsInVirtualVoxelGrid;

        uint3 clampedHitCoordsInVirtualVoxelGrid = clamp(uint3(hitPosInVirtualVoxelGrid), brickMinCoordsInVirtualVoxelGrid, brickMaxCoordsInVirtualVoxelGrid);
        // Calculate the voxel unit coords from the difference between the actual hit position and the base of the voxel we hit.
        virtualVoxelBrickPos = hitPosInVirtualVoxelGrid - brickCoordsInVirtualVoxelGrid;
        voxelUnitCoords = hitPosInVirtualVoxelGrid - clampedHitCoordsInVirtualVoxelGrid;

        return calculateBrickTexelCoords(virtualBrickCoords, brickTexelCoords);
    }

    bool calculateBrickCoordsApprox(float3 pLocal, out uint3 virtualBrickCoords, out uint2 brickTexelCoords, out int3 brickLocalVoxelCoords, out float3 voxelUnitCoords)
    {
        float3 virtualGridPos = clamp(pLocal * virtualGridWidth, 0.0f, float(virtualGridWidth));

        uint3 virtualVoxelCoords = uint3(virtualGridPos);
        voxelUnitCoords = virtualGridPos - virtualVoxelCoords;
        brickLocalVoxelCoords = virtualVoxelCoords % brickWidth;

        // Calculate the virtual brick coordinates.
        virtualBrickCoords = virtualVoxelCoords / brickWidth;

        return calculateBrickTexelCoords(virtualBrickCoords, brickTexelCoords);
    }

    bool calculateBrickTexelCoords(const uint3 virtualBrickCoords, out uint2 brickTexelCoords)
    {
        // Load the brick ID from the indirection buffer.
        uint brickID = indirectionBuffer[virtualBrickCoords];
        uint brickWidthInValues = brickWidth + 1;
        brickTexelCoords = uint2(brickID % bricksPerAxis.x, brickID / bricksPerAxis.x) * uint2(brickWidthInValues * brickWidthInValues, brickWidthInValues);

        return brickID != UINT32_MAX;
    }

    void loadVoxelValues(uint2 brickTexelCoords, int3 brickLocalVoxelCoords, out float4 values0xx, out float4 values1xx)
    {
        uint brickWidthInValues = brickWidth + 1;
        uint2 voxelPos = brickTexelCoords + uint2(brickLocalVoxelCoords.x + brickLocalVoxelCoords.z * brickWidthInValues, brickLocalVoxelCoords.y);
        float2 texCoords = (voxelPos + 1.0f) / brickTextureDimensions;

#if USE_GATHER
        float4 valuesxx0 = bricks.Gather(sampler, texCoords);
        float4 valuesxx1 = bricks.Gather(sampler, texCoords + float2(float(brickWidthInValues) / brickTextureDimensions.x, 0.0f));
        values0xx = float4(valuesxx0.w, valuesxx1.w, valuesxx0.x, valuesxx1.x);
        values1xx = float4(valuesxx0.z, valuesxx1.z, valuesxx0.y, valuesxx1.y);
#else
        values0xx = float4(bricks[voxelPos + uint2(0, 0)],
                           bricks[voxelPos + uint2(brickWidthInValues + 0, 0)],
                           bricks[voxelPos + uint2(0, 1)],
                           bricks[voxelPos + uint2(brickWidthInValues + 0, 1)]);
        values1xx = float4(bricks[voxelPos + uint2(1, 0)],
                           bricks[voxelPos + uint2(brickWidthInValues + 1, 0)],
                           bricks[voxelPos + uint2(1, 1)],
                           bricks[voxelPos + uint2(brickWidthInValues + 1, 1)]);
#endif
    }

    bool intersectSDF(const float3 rayOrigin, const float3 rayDir, const float tMin, const float tMax, const uint primitiveID, out float t, out SDFGridHitData hitData)
    {
        t = {};
        hitData = {};

        // Normalize ray direction.
        float dirLength = length(rayDir);
        float inverseDirLength = 1.0f / dirLength;
        float3 d = rayDir * inverseDirLength;

        // Clamp direction to epsilon to avoid division by zero.
        d.x = abs(d.x) < FLT_EPSILON ? (d.x < 0.0f ? -FLT_EPSILON : FLT_EPSILON) : d.x;
        d.y = abs(d.y) < FLT_EPSILON ? (d.y < 0.0f ? -FLT_EPSILON : FLT_EPSILON) : d.y;
        d.z = abs(d.z) < FLT_EPSILON ? (d.z < 0.0f ? -FLT_EPSILON : FLT_EPSILON) : d.z;

        // Load the intersected brick AABB.
        AABB aabb = aabbs[primitiveID];

        float2 nearFar;
        if (!intersectRayAABB(rayOrigin, d, aabb.minPoint, aabb.maxPoint, nearFar))
        {
            return false;
        }
        // Add 0.5f to origin so that it is in [0, 1] instead of [-0.5, 0.5].
        float3 rayOrigLocal = rayOrigin + 0.5f;

        // Compute min and max local position to avoid rounding errors.
        float3 brickMin = aabb.minPoint + 0.5f;
        float3 brickMax = aabb.maxPoint + 0.5f;

        // Set up t, o, and tVoxelMax.
        t = max(tMin * dirLength, nearFar.x);
        const float tVoxelMax = (min(tMax * dirLength, nearFar.y) - t) * virtualGridWidth;

        // Check that the ray segment overlaps the AABB.
        if (tVoxelMax < 0.0f) return false;

        const float3 o = rayOrigLocal + t * d;

        // Find current brick coords.
        uint3 virtualBrickCoords;
        uint2 brickTexelCoords;
        float3 virtualVoxelBrickPos;
        float3 voxelUnitCoords;
        int3 brickLocalMinCoords;
        int3 brickLocalMaxCoords;
        if (!calculateBrickCoords(o, brickMin, brickMax, virtualBrickCoords, brickTexelCoords, virtualVoxelBrickPos, voxelUnitCoords, brickLocalMinCoords, brickLocalMaxCoords))
            return false;

        int3 brickLocalVoxelCoords = clamp(int3(virtualVoxelBrickPos), brickLocalMinCoords, brickLocalMaxCoords);

        int3 stepSign;
        stepSign.x = d.x < 0.0f ? -1 : 1;
        stepSign.y = d.y < 0.0f ? -1 : 1;
        stepSign.z = d.z < 0.0f ? -1 : 1;

        float voxelWidth = 1.0f / virtualGridWidth;
        float normalizedVoxelDiagonal = sqrt(3.0f * voxelWidth * voxelWidth) / normalizationFactor;
        float3 invDir = 1.0f / d;
        float3 tDeltas = abs(invDir);
        float3 tVoxels = (step(float3(0.0f), d) - voxelUnitCoords) * invDir;
        int3 voxelIncrementation;
        float tLocal = 0.0f;
        float tLocalNext = min(min(tVoxels.x, tVoxels.y), tVoxels.z);

        // If the ray origin is inside the brick and we intersect with a voxel on the first iteration then we are inside that voxel.
        bool origInsideAABB = all(rayOrigLocal >= brickMin && rayOrigLocal <= brickMax);
        // DDA to surface containing voxel.
        while (all(brickLocalVoxelCoords >= brickLocalMinCoords) && all(brickLocalVoxelCoords <= brickLocalMaxCoords) && tLocal <= tVoxelMax)
        {
            // Load the voxel corner values.
            float4 values0xx;
            float4 values1xx;
            loadVoxelValues(brickTexelCoords, brickLocalVoxelCoords, values0xx, values1xx);

            // Todo: Optimize this check, valid bit per voxel?.
            if (SDFVoxelCommon::containsSurface(values0xx, values1xx))
            {
                float3 brickLocalHitP = virtualVoxelBrickPos + tLocal * d;                // The hit position in the local space of the virtual brick, i.e., [0, hitBrickWidthInVoxels).
                float3 voxelHitP = brickLocalHitP - brickLocalVoxelCoords;               // The hit position in the local space of the voxel, i.e., [0, 1).

                float tVoxelLocal = 0.0f;
                const float tVoxelLocalMax = kSnormErrorTolerance + min(tVoxelMax, tLocalNext) - tLocal;
                if (SDFVoxelHitUtils::intersectSDFVoxel(voxelHitP, d, origInsideAABB, values0xx, values1xx, tVoxelLocalMax, kSolverMaxStepCount, tVoxelLocal))
                {
                    t += (tLocal + tVoxelLocal) * voxelWidth;
                    t *= inverseDirLength;

                    hitData.primitiveID = primitiveID;
                    hitData.hitT = t;
                    return true;
                }
            }

            voxelIncrementation.x = int(tVoxels.x <= tVoxels.y && tVoxels.x <= tVoxels.z);
            voxelIncrementation.y = int(tVoxels.y <= tVoxels.x && tVoxels.y <= tVoxels.z);
            voxelIncrementation.z = int(tVoxels.z <= tVoxels.x && tVoxels.z <= tVoxels.y);

            tVoxels += tDeltas * voxelIncrementation;
            brickLocalVoxelCoords += voxelIncrementation * stepSign;
            tLocal = tLocalNext;
            tLocalNext = min(min(tVoxels.x, tVoxels.y), tVoxels.z);
            origInsideAABB = false;
        }

        return false;
    }

    bool intersectSDFAny(const float3 rayOrigin, const float3 rayDir, const float tMin, const float tMax, const uint primitiveID)
    {
        // Normalize ray direction.
        float dirLength = length(rayDir);
        float inverseDirLength = 1.0f / dirLength;
        float3 d = rayDir * inverseDirLength;

        // Clamp direction to epsilon to avoid division by zero.
        d.x = abs(d.x) < FLT_EPSILON ? (d.x < 0.0f ? -FLT_EPSILON : FLT_EPSILON) : d.x;
        d.y = abs(d.y) < FLT_EPSILON ? (d.y < 0.0f ? -FLT_EPSILON : FLT_EPSILON) : d.y;
        d.z = abs(d.z) < FLT_EPSILON ? (d.z < 0.0f ? -FLT_EPSILON : FLT_EPSILON) : d.z;

        // Load the intersected brick AABB.
        AABB aabb = aabbs[primitiveID];

        float2 nearFar;
        if (!intersectRayAABB(rayOrigin, d, aabb.minPoint, aabb.maxPoint, nearFar))
            return false;

        // Add 0.5f to origin so that it is in [0, 1] instead of [-0.5, 0.5].
        float3 rayOrigLocal = rayOrigin + 0.5f;

        // Compute min and max local position to avoid rounding errors.
        float3 brickMin = aabb.minPoint + 0.5f;
        float3 brickMax = aabb.maxPoint + 0.5f;

        // Set up t, o, and tVoxelMax.
        float t = max(tMin * dirLength, nearFar.x);
        const float tVoxelMax = (min(tMax * dirLength, nearFar.y) - t) * virtualGridWidth;

        // Check that the ray segment overlaps the AABB.
        if (tVoxelMax < 0.0f) return false;

        const float3 o = clamp(rayOrigLocal + t * d, brickMin, brickMax);

        // Find current brick coords.
        uint3 virtualBrickCoords;
        uint2 brickTexelCoords;
        float3 virtualVoxelBrickPos;
        float3 voxelUnitCoords;
        int3 brickLocalMinCoords;
        int3 brickLocalMaxCoords;
        if (!calculateBrickCoords(o, brickMin, brickMax, virtualBrickCoords, brickTexelCoords, virtualVoxelBrickPos, voxelUnitCoords, brickLocalMinCoords, brickLocalMaxCoords))
            return false;

        int3 brickLocalVoxelCoords = int3(virtualVoxelBrickPos);

        int3 stepSign;
        stepSign.x = d.x < 0.0f ? -1 : 1;
        stepSign.y = d.y < 0.0f ? -1 : 1;
        stepSign.z = d.z < 0.0f ? -1 : 1;

        float voxelWidth = 1.0f / virtualGridWidth;
        float normalizedVoxelDiagonal = sqrt(3.0f * voxelWidth * voxelWidth) / normalizationFactor;
        float3 invDir = 1.0f / d;
        float3 tDeltas = abs(invDir);
        float3 tVoxels = (step(float3(0.0f), d) - voxelUnitCoords) * invDir;
        int3 voxelIncrementation;
        float tLocal = 0.0f;
        float tLocalNext = min(min(tVoxels.x, tVoxels.y), tVoxels.z);

        // If the ray origin is inside the brick and we intersect with a voxel on the first iteration then we are inside that voxel.
        bool origInsideAABB = all(rayOrigLocal >= brickMin && rayOrigLocal <= brickMax);

        // DDA to surface containing voxel.
        while (all(brickLocalVoxelCoords >= brickLocalMinCoords) && all(brickLocalVoxelCoords <= brickLocalMaxCoords) && tLocal <= tVoxelMax)
        {
            // Load the voxel corner values.
            float4 values0xx;
            float4 values1xx;
            loadVoxelValues(brickTexelCoords, brickLocalVoxelCoords, values0xx, values1xx);

            // Todo: Optimize this check, valid bit per voxel?.
            if (SDFVoxelCommon::containsSurface(values0xx, values1xx))
            {
                float3 brickLocalHitP = virtualVoxelBrickPos + tLocal * d;                // The hit position in the local space of the virtual brick, i.e., [0, hitBrickWidthInVoxels).
                float3 voxelHitP = brickLocalHitP - brickLocalVoxelCoords;               // The hit position in the local space of the voxel, i.e., [0, 1).

                const float tVoxelLocalMax = kSnormErrorTolerance + min(tVoxelMax, tLocalNext) - tLocal;
                if (SDFVoxelHitUtils::intersectSDFVoxelAny(voxelHitP, d, origInsideAABB, values0xx, values1xx, tVoxelLocalMax, kSolverMaxStepCount))
                {
                    return true;
                }
            }

            voxelIncrementation.x = uint(tVoxels.x <= tVoxels.y && tVoxels.x <= tVoxels.z);
            voxelIncrementation.y = uint(tVoxels.y <= tVoxels.x && tVoxels.y <= tVoxels.z);
            voxelIncrementation.z = uint(tVoxels.z <= tVoxels.x && tVoxels.z <= tVoxels.y);

            tVoxels += tDeltas * voxelIncrementation;
            brickLocalVoxelCoords += voxelIncrementation * stepSign;
            tLocal = tLocalNext;
            tLocalNext = min(min(tVoxels.x, tVoxels.y), tVoxels.z);
            origInsideAABB = false;
        }

        return false;
    }

    void decodeHit(const SDFGridHitData hitData, const float3 hitPosition, out float3 gradient)
    {
        // Add 0.5f to hitPosition so that it is in [0, 1] instead of [-0.5, 0.5].
        float3 hitPosLocal = hitPosition + 0.5f;

#if SCENE_SDF_GRADIENT_EVALUATION_METHOD == SCENE_SDF_GRADIENT_NUMERIC_DISCONTINUOUS
        // Find current brick coords.
        uint3 virtualBrickCoords;
        uint2 brickTexelCoords;
        int3 brickLocalVoxelCoords;
        float3 voxelUnitCoords;
        calculateBrickCoordsApprox(hitPosLocal, virtualBrickCoords, brickTexelCoords, brickLocalVoxelCoords, voxelUnitCoords);

        float4 values0xx;
        float4 values1xx;
        loadVoxelValues(brickTexelCoords, brickLocalVoxelCoords, values0xx, values1xx);

        const float offset = 0.2f / virtualGridWidth;
        gradient = SDFVoxelHitUtils::computeNumericGradient(voxelUnitCoords, offset, values0xx, values1xx);
#endif
        gradient *= normalizationFactor;
    }

};

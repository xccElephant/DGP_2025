#pragma once

#include <pxr/base/tf/weakPtr.h>
#include <pxr/usd/usd/notice.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>

#include <mutex>

#include "api.h"
#include "stage/stage.hpp"
#include "stage_listener/api.h"

USTC_CG_NAMESPACE_OPEN_SCOPE

class STAGE_LISTENER_API StageListener : public pxr::TfWeakBase {
   public:
    using DirtyPrimSet = std::unordered_set<pxr::SdfPath, pxr::SdfPath::Hash>;
    explicit StageListener(Stage* stage) : stage_(stage)
    {
        // 检查stage是否有效
        if (!stage_ || !stage_->get_usd_stage()) {
            throw std::runtime_error("Invalid stage pointer");
        }

        pxr::UsdStageRefPtr stage_ref_ptr(stage_->get_usd_stage());

        // 注册监听场景内容变化（Prim添加/删除）
        stageContentsChangedKey_ = pxr::TfNotice::Register(
            pxr::TfCreateWeakPtr(this),
            &StageListener::OnStageContentsChanged,
            stage_ref_ptr);

        // 注册监听属性变化
        objectsChangedKey_ = pxr::TfNotice::Register(
            pxr::TfCreateWeakPtr(this),
            &StageListener::OnObjectsChanged,
            stage_ref_ptr);
    }

    std::mutex& GetMutex()
    {
        return mutex_;
    }

    void GetDirtyPrims(DirtyPrimSet& outDirtyPrims)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        outDirtyPrims = std::move(dirtyPrims_);
        dirtyPrims_.clear();  // 确保原容器为空
    }

   private:
    // 处理Prim结构变化（添加/删除）
    void OnStageContentsChanged(
        const pxr::UsdNotice::StageContentsChanged& notice)
    {
        if (!stage_ || !stage_->get_usd_stage()) {
            return;
        }

        // 记录需要处理的Prim路径
        std::lock_guard<std::mutex> lock(mutex_);
        auto stage_ptr = stage_->get_usd_stage();
        for (const pxr::UsdPrim& prim : stage_ptr->Traverse()) {
            // 标记新增或删除的Prim
            dirtyPrims_.insert(prim.GetPath());
        }
    }

    // 处理属性变化（如变换、几何数据）
    void OnObjectsChanged(const pxr::UsdNotice::ObjectsChanged& notice)
    {
        if (!stage_) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        for (const pxr::SdfPath& path : notice.GetChangedInfoOnlyPaths()) {
            // 标记属性变化的Prim路径
            dirtyPrims_.insert(path.GetPrimPath());
        }
    }

    Stage* stage_;
    std::mutex mutex_;
    DirtyPrimSet dirtyPrims_;  // 脏Prim路径集合
    pxr::TfNotice::Key stageContentsChangedKey_, objectsChangedKey_;
};

USTC_CG_NAMESPACE_CLOSE_SCOPE
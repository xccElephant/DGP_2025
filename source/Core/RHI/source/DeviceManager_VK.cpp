/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/*
License for glfw

Copyright (c) 2002-2006 Marcus Geelnard

Copyright (c) 2006-2019 Camilla Lowy

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would
   be appreciated but is not required.

2. Altered source versions must be plainly marked as such, and must not
   be misrepresented as being the original software.

3. This notice may not be removed or altered from any source
   distribution.
*/

#include <RHI/DeviceManager/DeviceManager.h>
#include <nvrhi/validation.h>
#include <nvrhi/vulkan.h>

#include <queue>
#include <string>
#include <unordered_set>

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <sstream>
#include <vulkan/vulkan.hpp>

USTC_CG_NAMESPACE_OPEN_SCOPE

class DeviceManager_VK : public DeviceManager {
   public:
    [[nodiscard]] nvrhi::IDevice* GetDevice() const override
    {
        if (m_ValidationLayer)
            return m_ValidationLayer;

        return m_NvrhiDevice;
    }

    [[nodiscard]] nvrhi::GraphicsAPI GetGraphicsAPI() const override
    {
        return nvrhi::GraphicsAPI::VULKAN;
    }

    bool EnumerateAdapters(std::vector<AdapterInfo>& outAdapters) override;

   protected:
    bool CreateInstanceInternal() override;
    bool CreateDevice() override;
    bool CreateSwapChain() override;
    void DestroyDeviceAndSwapChain() override;

    void ResizeSwapChain() override
    {
        if (m_VulkanDevice) {
            destroySwapChain();
            createSwapChain();
        }
    }

    nvrhi::ITexture* GetCurrentBackBuffer() override
    {
        return m_SwapChainImages[m_SwapChainIndex].rhiHandle;
    }
    nvrhi::ITexture* GetBackBuffer(uint32_t index) override
    {
        if (index < m_SwapChainImages.size())
            return m_SwapChainImages[index].rhiHandle;
        return nullptr;
    }
    uint32_t GetCurrentBackBufferIndex() override
    {
        return m_SwapChainIndex;
    }
    uint32_t GetBackBufferCount() override
    {
        return uint32_t(m_SwapChainImages.size());
    }

    bool BeginFrame() override;
    bool Present() override;

    const char* GetRendererString() const override
    {
        return m_RendererString.c_str();
    }

    bool IsVulkanInstanceExtensionEnabled(
        const char* extensionName) const override
    {
        return enabledExtensions.instance.find(extensionName) !=
               enabledExtensions.instance.end();
    }

    bool IsVulkanDeviceExtensionEnabled(
        const char* extensionName) const override
    {
        return enabledExtensions.device.find(extensionName) !=
               enabledExtensions.device.end();
    }

    bool IsVulkanLayerEnabled(const char* layerName) const override
    {
        return enabledExtensions.layers.find(layerName) !=
               enabledExtensions.layers.end();
    }

    void GetEnabledVulkanInstanceExtensions(
        std::vector<std::string>& extensions) const override
    {
        for (const auto& ext : enabledExtensions.instance)
            extensions.push_back(ext);
    }

    void GetEnabledVulkanDeviceExtensions(
        std::vector<std::string>& extensions) const override
    {
        for (const auto& ext : enabledExtensions.device)
            extensions.push_back(ext);
    }

    void GetEnabledVulkanLayers(std::vector<std::string>& layers) const override
    {
        for (const auto& ext : enabledExtensions.layers)
            layers.push_back(ext);
    }

   private:
    bool createInstance();
    bool createWindowSurface();
    void installDebugCallback();
    bool pickPhysicalDevice();
    bool findQueueFamilies(vk::PhysicalDevice physicalDevice);
    bool createDevice();
    bool createSwapChain();
    void destroySwapChain();

    struct VulkanExtensionSet {
        std::unordered_set<std::string> instance;
        std::unordered_set<std::string> layers;
        std::unordered_set<std::string> device;
    };

    // minimal set of required extensions
    VulkanExtensionSet enabledExtensions = {
        // instance
        { VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME },
        // layers
        {},
        // device
        { VK_KHR_MAINTENANCE1_EXTENSION_NAME },
    };

    // optional extensions
    VulkanExtensionSet optionalExtensions = {
        // instance
        { VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME,
          VK_EXT_DEBUG_UTILS_EXTENSION_NAME },
        // layers
        {},
        // device
        {
            VK_EXT_DEBUG_MARKER_EXTENSION_NAME,
            VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            VK_NV_MESH_SHADER_EXTENSION_NAME,
            VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME,
            VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
            VK_KHR_MAINTENANCE_4_EXTENSION_NAME,
            VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME,
#if USTC_CG_WITH_AFTERMATH
            VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME,
            VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME,
#endif
        },
    };

    std::unordered_set<std::string> m_RayTracingExtensions = {
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME
    };

    std::string m_RendererString;

    vk::Instance m_VulkanInstance;
    vk::DebugReportCallbackEXT m_DebugReportCallback;

    vk::PhysicalDevice m_VulkanPhysicalDevice;
    int m_GraphicsQueueFamily = -1;
    int m_ComputeQueueFamily = -1;
    int m_TransferQueueFamily = -1;
    int m_PresentQueueFamily = -1;

    vk::Device m_VulkanDevice;
    vk::Queue m_GraphicsQueue;
    vk::Queue m_ComputeQueue;
    vk::Queue m_TransferQueue;
    vk::Queue m_PresentQueue;

    vk::SurfaceKHR m_WindowSurface;

    vk::SurfaceFormatKHR m_SwapChainFormat;
    vk::SwapchainKHR m_SwapChain;
    bool m_SwapChainMutableFormatSupported = false;

    struct SwapChainImage {
        vk::Image image;
        nvrhi::TextureHandle rhiHandle;
    };

    std::vector<SwapChainImage> m_SwapChainImages;
    uint32_t m_SwapChainIndex = uint32_t(-1);

    nvrhi::vulkan::DeviceHandle m_NvrhiDevice;
    nvrhi::DeviceHandle m_ValidationLayer;

    nvrhi::CommandListHandle m_BarrierCommandList;
    std::vector<vk::Semaphore> m_PresentSemaphores;
    uint32_t m_PresentSemaphoreIndex = 0;

    std::queue<nvrhi::EventQueryHandle> m_FramesInFlight;
    std::vector<nvrhi::EventQueryHandle> m_QueryPool;

    bool m_BufferDeviceAddressSupported = false;

    vk::DynamicLoader m_dynamicLoader;

   private:
    static VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugCallback(
        VkDebugReportFlagsEXT flags,
        VkDebugReportObjectTypeEXT objType,
        uint64_t obj,
        size_t location,
        int32_t code,
        const char* layerPrefix,
        const char* msg,
        void* userData)
    {
        const DeviceManager_VK* manager = (const DeviceManager_VK*)userData;

        if (manager) {
            const auto& ignored =
                manager->m_DeviceParams.ignoredVulkanValidationMessageLocations;
            const auto found =
                std::find(ignored.begin(), ignored.end(), location);
            if (found != ignored.end())
                return VK_FALSE;
        }

        log::warning(
            "[Vulkan: location=0x%zx code=%d, layerPrefix='%s'] %s",
            location,
            code,
            layerPrefix,
            msg);

        return VK_FALSE;
    }
};

static std::vector<const char*> stringSetToVector(
    const std::unordered_set<std::string>& set)
{
    std::vector<const char*> ret;
    for (const auto& s : set) {
        ret.push_back(s.c_str());
    }

    return ret;
}

template<typename T>
static std::vector<T> setToVector(const std::unordered_set<T>& set)
{
    std::vector<T> ret;
    for (const auto& s : set) {
        ret.push_back(s);
    }

    return ret;
}

bool DeviceManager_VK::createInstance()
{
    if (!m_DeviceParams.headlessDevice) {
        if (!glfwVulkanSupported()) {
            log::error(
                "GLFW reports that Vulkan is not supported. Perhaps missing a "
                "call to glfwInit()?");
            return false;
        }

        // add any extensions required by GLFW
        uint32_t glfwExtCount;
        const char** glfwExt = glfwGetRequiredInstanceExtensions(&glfwExtCount);
        assert(glfwExt);

        for (uint32_t i = 0; i < glfwExtCount; i++) {
            enabledExtensions.instance.insert(std::string(glfwExt[i]));
        }
    }

    // add instance extensions requested by the user
    for (const std::string& name :
         m_DeviceParams.requiredVulkanInstanceExtensions) {
        enabledExtensions.instance.insert(name);
    }
    for (const std::string& name :
         m_DeviceParams.optionalVulkanInstanceExtensions) {
        optionalExtensions.instance.insert(name);
    }

    // add layers requested by the user
    for (const std::string& name : m_DeviceParams.requiredVulkanLayers) {
        enabledExtensions.layers.insert(name);
    }
    for (const std::string& name : m_DeviceParams.optionalVulkanLayers) {
        optionalExtensions.layers.insert(name);
    }

    std::unordered_set<std::string> requiredExtensions =
        enabledExtensions.instance;

    // figure out which optional extensions are supported
    for (const auto& instanceExt : vk::enumerateInstanceExtensionProperties()) {
        const std::string name = instanceExt.extensionName;
        if (optionalExtensions.instance.find(name) !=
            optionalExtensions.instance.end()) {
            enabledExtensions.instance.insert(name);
        }

        requiredExtensions.erase(name);
    }

    if (!requiredExtensions.empty()) {
        std::stringstream ss;
        ss << "Cannot create a Vulkan instance because the following required "
              "extension(s) are not supported:";
        for (const auto& ext : requiredExtensions)
            ss << std::endl << "  - " << ext;

        log::error("%s", ss.str().c_str());
        return false;
    }

    log::message(
        m_DeviceParams.infoLogSeverity, "Enabled Vulkan instance extensions:");
    for (const auto& ext : enabledExtensions.instance) {
        log::message(m_DeviceParams.infoLogSeverity, "    %s", ext.c_str());
    }

    std::unordered_set<std::string> requiredLayers = enabledExtensions.layers;

    for (const auto& layer : vk::enumerateInstanceLayerProperties()) {
        const std::string name = layer.layerName;
        if (optionalExtensions.layers.find(name) !=
            optionalExtensions.layers.end()) {
            enabledExtensions.layers.insert(name);
        }

        requiredLayers.erase(name);
    }

    if (!requiredLayers.empty()) {
        std::stringstream ss;
        ss << "Cannot create a Vulkan instance because the following required "
              "layer(s) are not supported:";
        for (const auto& ext : requiredLayers)
            ss << std::endl << "  - " << ext;

        log::error("%s", ss.str().c_str());
        return false;
    }

    log::message(m_DeviceParams.infoLogSeverity, "Enabled Vulkan layers:");
    for (const auto& layer : enabledExtensions.layers) {
        log::message(m_DeviceParams.infoLogSeverity, "    %s", layer.c_str());
    }

    auto instanceExtVec = stringSetToVector(enabledExtensions.instance);
    auto layerVec = stringSetToVector(enabledExtensions.layers);

    auto applicationInfo = vk::ApplicationInfo();

    // Query the Vulkan API version supported on the system to make sure we use
    // at least 1.3 when that's present.
    vk::Result res = vk::enumerateInstanceVersion(&applicationInfo.apiVersion);

    if (res != vk::Result::eSuccess) {
        log::error(
            "Call to vkEnumerateInstanceVersion failed, error code = %s",
            nvrhi::vulkan::resultToString(VkResult(res)));
        return false;
    }

    const uint32_t minimumVulkanVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);

    // Check if the Vulkan API version is sufficient.
    if (applicationInfo.apiVersion < minimumVulkanVersion) {
        log::error(
            "The Vulkan API version supported on the system (%d.%d.%d) is too "
            "low, at least %d.%d.%d is required.",
            VK_API_VERSION_MAJOR(applicationInfo.apiVersion),
            VK_API_VERSION_MINOR(applicationInfo.apiVersion),
            VK_API_VERSION_PATCH(applicationInfo.apiVersion),
            VK_API_VERSION_MAJOR(minimumVulkanVersion),
            VK_API_VERSION_MINOR(minimumVulkanVersion),
            VK_API_VERSION_PATCH(minimumVulkanVersion));
        return false;
    }

    // Spec says: A non-zero variant indicates the API is a variant of the
    // Vulkan API and applications will typically need to be modified to run
    // against it.
    if (VK_API_VERSION_VARIANT(applicationInfo.apiVersion) != 0) {
        log::error(
            "The Vulkan API supported on the system uses an unexpected "
            "variant: %d.",
            VK_API_VERSION_VARIANT(applicationInfo.apiVersion));
        return false;
    }

    // Create the vulkan instance
    vk::InstanceCreateInfo info =
        vk::InstanceCreateInfo()
            .setEnabledLayerCount(uint32_t(layerVec.size()))
            .setPpEnabledLayerNames(layerVec.data())
            .setEnabledExtensionCount(uint32_t(instanceExtVec.size()))
            .setPpEnabledExtensionNames(instanceExtVec.data())
            .setPApplicationInfo(&applicationInfo);

    res = vk::createInstance(&info, nullptr, &m_VulkanInstance);
    if (res != vk::Result::eSuccess) {
        log::error(
            "Failed to create a Vulkan instance, error code = %s",
            nvrhi::vulkan::resultToString(VkResult(res)));
        return false;
    }

    VULKAN_HPP_DEFAULT_DISPATCHER.init(m_VulkanInstance);

    return true;
}

void DeviceManager_VK::installDebugCallback()
{
    auto info = vk::DebugReportCallbackCreateInfoEXT()
                    .setFlags(
                        vk::DebugReportFlagBitsEXT::eError |
                        vk::DebugReportFlagBitsEXT::eWarning |
                        //   vk::DebugReportFlagBitsEXT::eInformation |
                        vk::DebugReportFlagBitsEXT::ePerformanceWarning)
                    .setPfnCallback(vulkanDebugCallback)
                    .setPUserData(this);

    vk::Result res = m_VulkanInstance.createDebugReportCallbackEXT(
        &info, nullptr, &m_DebugReportCallback);
    assert(res == vk::Result::eSuccess);
}

bool DeviceManager_VK::pickPhysicalDevice()
{
    VkFormat requestedFormat =
        nvrhi::vulkan::convertFormat(m_DeviceParams.swapChainFormat);
    vk::Extent2D requestedExtent(
        m_DeviceParams.backBufferWidth, m_DeviceParams.backBufferHeight);

    auto devices = m_VulkanInstance.enumeratePhysicalDevices();

    int firstDevice = 0;
    int lastDevice = int(devices.size()) - 1;
    if (m_DeviceParams.adapterIndex >= 0) {
        if (m_DeviceParams.adapterIndex > lastDevice) {
            log::error(
                "The specified Vulkan physical device %d does not exist.",
                m_DeviceParams.adapterIndex);
            return false;
        }
        firstDevice = m_DeviceParams.adapterIndex;
        lastDevice = m_DeviceParams.adapterIndex;
    }

    // Start building an error message in case we cannot find a device.
    std::stringstream errorStream;
    errorStream << "Cannot find a Vulkan device that supports all the required "
                   "extensions and properties.";

    // build a list of GPUs
    std::vector<vk::PhysicalDevice> discreteGPUs;
    std::vector<vk::PhysicalDevice> otherGPUs;
    for (int deviceIndex = firstDevice; deviceIndex <= lastDevice;
         ++deviceIndex) {
        vk::PhysicalDevice const& dev = devices[deviceIndex];
        vk::PhysicalDeviceProperties prop = dev.getProperties();

        errorStream << std::endl << prop.deviceName.data() << ":";

        // check that all required device extensions are present
        std::unordered_set<std::string> requiredExtensions =
            enabledExtensions.device;
        auto deviceExtensions = dev.enumerateDeviceExtensionProperties();
        for (const auto& ext : deviceExtensions) {
            requiredExtensions.erase(std::string(ext.extensionName.data()));
        }

        bool deviceIsGood = true;

        if (!requiredExtensions.empty()) {
            // device is missing one or more required extensions
            for (const auto& ext : requiredExtensions) {
                errorStream << std::endl << "  - missing " << ext;
            }
            deviceIsGood = false;
        }

        auto deviceFeatures = dev.getFeatures();
        if (!deviceFeatures.samplerAnisotropy) {
            // device is a toaster oven
            errorStream << std::endl
                        << "  - does not support samplerAnisotropy";
            deviceIsGood = false;
        }
        if (!deviceFeatures.textureCompressionBC) {
            errorStream << std::endl
                        << "  - does not support textureCompressionBC";
            deviceIsGood = false;
        }

        if (!findQueueFamilies(dev)) {
            // device doesn't have all the queue families we need
            errorStream << std::endl
                        << "  - does not support the necessary queue types";
            deviceIsGood = false;
        }

        if (deviceIsGood && m_WindowSurface) {
            bool surfaceSupported =
                dev.getSurfaceSupportKHR(m_PresentQueueFamily, m_WindowSurface);
            if (!surfaceSupported) {
                errorStream << std::endl
                            << "  - does not support the window surface";
                deviceIsGood = false;
            }
            else {
                // check that this device supports our intended swap chain
                // creation parameters
                auto surfaceCaps =
                    dev.getSurfaceCapabilitiesKHR(m_WindowSurface);
                auto surfaceFmts = dev.getSurfaceFormatsKHR(m_WindowSurface);

                if (surfaceCaps.minImageCount >
                        m_DeviceParams.swapChainBufferCount ||
                    (surfaceCaps.maxImageCount <
                         m_DeviceParams.swapChainBufferCount &&
                     surfaceCaps.maxImageCount > 0)) {
                    errorStream << std::endl
                                << "  - cannot support the requested swap "
                                   "chain image count:";
                    errorStream
                        << " requested " << m_DeviceParams.swapChainBufferCount
                        << ", available " << surfaceCaps.minImageCount << " - "
                        << surfaceCaps.maxImageCount;
                    deviceIsGood = false;
                }

                if (surfaceCaps.minImageExtent.width > requestedExtent.width ||
                    surfaceCaps.minImageExtent.height >
                        requestedExtent.height ||
                    surfaceCaps.maxImageExtent.width < requestedExtent.width ||
                    surfaceCaps.maxImageExtent.height <
                        requestedExtent.height) {
                    errorStream
                        << std::endl
                        << "  - cannot support the requested swap chain size:";
                    errorStream << " requested " << requestedExtent.width << "x"
                                << requestedExtent.height << ", ";
                    errorStream << " available "
                                << surfaceCaps.minImageExtent.width << "x"
                                << surfaceCaps.minImageExtent.height;
                    errorStream << " - " << surfaceCaps.maxImageExtent.width
                                << "x" << surfaceCaps.maxImageExtent.height;
                    deviceIsGood = false;
                }

                bool surfaceFormatPresent = false;
                for (const vk::SurfaceFormatKHR& surfaceFmt : surfaceFmts) {
                    if (surfaceFmt.format == vk::Format(requestedFormat)) {
                        surfaceFormatPresent = true;
                        break;
                    }
                }

                if (!surfaceFormatPresent) {
                    // can't create a swap chain using the format requested
                    errorStream << std::endl
                                << "  - does not support the requested swap "
                                   "chain format";
                    deviceIsGood = false;
                }

                // check that we can present from the graphics queue
                uint32_t canPresent = dev.getSurfaceSupportKHR(
                    m_GraphicsQueueFamily, m_WindowSurface);
                if (!canPresent) {
                    errorStream << std::endl << "  - cannot present";
                    deviceIsGood = false;
                }
            }
        }

        if (!deviceIsGood)
            continue;

        if (prop.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
            discreteGPUs.push_back(dev);
        }
        else {
            otherGPUs.push_back(dev);
        }
    }

    // pick the first discrete GPU if it exists, otherwise the first integrated
    // GPU
    if (!discreteGPUs.empty()) {
        m_VulkanPhysicalDevice = discreteGPUs[0];
        return true;
    }

    if (!otherGPUs.empty()) {
        m_VulkanPhysicalDevice = otherGPUs[0];
        return true;
    }

    log::error("%s", errorStream.str().c_str());

    return false;
}

bool DeviceManager_VK::findQueueFamilies(vk::PhysicalDevice physicalDevice)
{
    auto props = physicalDevice.getQueueFamilyProperties();

    for (int i = 0; i < int(props.size()); i++) {
        const auto& queueFamily = props[i];

        if (m_GraphicsQueueFamily == -1) {
            if (queueFamily.queueCount > 0 &&
                (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)) {
                m_GraphicsQueueFamily = i;
            }
        }

        if (m_ComputeQueueFamily == -1) {
            if (queueFamily.queueCount > 0 &&
                (queueFamily.queueFlags & vk::QueueFlagBits::eCompute) &&
                !(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)) {
                m_ComputeQueueFamily = i;
            }
        }

        if (m_TransferQueueFamily == -1) {
            if (queueFamily.queueCount > 0 &&
                (queueFamily.queueFlags & vk::QueueFlagBits::eTransfer) &&
                !(queueFamily.queueFlags & vk::QueueFlagBits::eCompute) &&
                !(queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)) {
                m_TransferQueueFamily = i;
            }
        }

        if (m_PresentQueueFamily == -1) {
            if (queueFamily.queueCount > 0 &&
                glfwGetPhysicalDevicePresentationSupport(
                    m_VulkanInstance, physicalDevice, i)) {
                m_PresentQueueFamily = i;
            }
        }
    }

    if (m_GraphicsQueueFamily == -1 ||
        (m_PresentQueueFamily == -1 && !m_DeviceParams.headlessDevice) ||
        (m_ComputeQueueFamily == -1 && m_DeviceParams.enableComputeQueue) ||
        (m_TransferQueueFamily == -1 && m_DeviceParams.enableCopyQueue)) {
        return false;
    }

    return true;
}

bool DeviceManager_VK::createDevice()
{
    // figure out which optional extensions are supported
    auto deviceExtensions =
        m_VulkanPhysicalDevice.enumerateDeviceExtensionProperties();
    for (const auto& ext : deviceExtensions) {
        const std::string name = ext.extensionName;
        if (optionalExtensions.device.find(name) !=
            optionalExtensions.device.end()) {
            if (name == VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME &&
                m_DeviceParams.headlessDevice)
                continue;

            enabledExtensions.device.insert(name);
        }

        if (m_DeviceParams.enableRayTracingExtensions &&
            m_RayTracingExtensions.find(name) != m_RayTracingExtensions.end()) {
            enabledExtensions.device.insert(name);
        }
    }

    if (!m_DeviceParams.headlessDevice) {
        enabledExtensions.device.insert(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    const vk::PhysicalDeviceProperties physicalDeviceProperties =
        m_VulkanPhysicalDevice.getProperties();
    m_RendererString = std::string(physicalDeviceProperties.deviceName.data());

    bool accelStructSupported = false;
    bool rayPipelineSupported = false;
    bool rayQuerySupported = false;
    bool meshletsSupported = false;
    bool vrsSupported = false;
    bool synchronization2Supported = false;
    bool maintenance4Supported = false;
    bool aftermathSupported = false;
    bool shaderAtomicSupported = false;

    log::message(
        m_DeviceParams.infoLogSeverity, "Enabled Vulkan device extensions:");
    for (const auto& ext : enabledExtensions.device) {
        log::message(m_DeviceParams.infoLogSeverity, "    %s", ext.c_str());

        if (ext == VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
            accelStructSupported = true;
        else if (ext == VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
            rayPipelineSupported = true;
        else if (ext == VK_KHR_RAY_QUERY_EXTENSION_NAME)
            rayQuerySupported = true;
        else if (ext == VK_NV_MESH_SHADER_EXTENSION_NAME)
            meshletsSupported = true;
        else if (ext == VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME)
            vrsSupported = true;
        else if (ext == VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)
            synchronization2Supported = true;
        else if (ext == VK_KHR_MAINTENANCE_4_EXTENSION_NAME)
            maintenance4Supported = true;
        else if (ext == VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME)
            m_SwapChainMutableFormatSupported = true;
        else if (ext == VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME)
            aftermathSupported = true;
        else if (ext == VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)
            shaderAtomicSupported = true;
    }

#define APPEND_EXTENSION(condition, desc) \
    if (condition) {                      \
        (desc).pNext = pNext;             \
        pNext = &(desc);                  \
    }  // NOLINT(cppcoreguidelines-macro-usage)
    void* pNext = nullptr;

    vk::PhysicalDeviceFeatures2 physicalDeviceFeatures2;
    // Determine support for Buffer Device Address, the Vulkan 1.2 way
    auto bufferDeviceAddressFeatures =
        vk::PhysicalDeviceBufferDeviceAddressFeatures();
    // Determine support for maintenance4
    auto maintenance4Features = vk::PhysicalDeviceMaintenance4Features();
    // Determine support for aftermath
    auto aftermathPhysicalFeatures =
        vk::PhysicalDeviceDiagnosticsConfigFeaturesNV();

    // Put the user-provided extension structure at the end of the chain
    pNext = m_DeviceParams.physicalDeviceFeatures2Extensions;
    APPEND_EXTENSION(true, bufferDeviceAddressFeatures);
    APPEND_EXTENSION(maintenance4Supported, maintenance4Features);
    APPEND_EXTENSION(aftermathSupported, aftermathPhysicalFeatures);

    physicalDeviceFeatures2.pNext = pNext;
    m_VulkanPhysicalDevice.getFeatures2(&physicalDeviceFeatures2);

    std::unordered_set<int> uniqueQueueFamilies = { m_GraphicsQueueFamily };

    if (!m_DeviceParams.headlessDevice)
        uniqueQueueFamilies.insert(m_PresentQueueFamily);

    if (m_DeviceParams.enableComputeQueue)
        uniqueQueueFamilies.insert(m_ComputeQueueFamily);

    if (m_DeviceParams.enableCopyQueue)
        uniqueQueueFamilies.insert(m_TransferQueueFamily);

    float priority = 1.f;
    std::vector<vk::DeviceQueueCreateInfo> queueDesc;
    queueDesc.reserve(uniqueQueueFamilies.size());
    for (int queueFamily : uniqueQueueFamilies) {
        queueDesc.push_back(vk::DeviceQueueCreateInfo()
                                .setQueueFamilyIndex(queueFamily)
                                .setQueueCount(1)
                                .setPQueuePriorities(&priority));
    }

    auto accelStructFeatures =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR()
            .setAccelerationStructure(true);
    auto rayPipelineFeatures = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR()
                                   .setRayTracingPipeline(true)
                                   .setRayTraversalPrimitiveCulling(true);
    auto rayQueryFeatures =
        vk::PhysicalDeviceRayQueryFeaturesKHR().setRayQuery(true);
    auto meshletFeatures = vk::PhysicalDeviceMeshShaderFeaturesNV()
                               .setTaskShader(true)
                               .setMeshShader(true);
    auto vrsFeatures = vk::PhysicalDeviceFragmentShadingRateFeaturesKHR()
                           .setPipelineFragmentShadingRate(true)
                           .setPrimitiveFragmentShadingRate(true)
                           .setAttachmentFragmentShadingRate(true);
    auto vulkan13features =
        vk::PhysicalDeviceVulkan13Features()
            .setSynchronization2(synchronization2Supported)
            .setMaintenance4(maintenance4Features.maintenance4);
    auto aftermathFeatures = vk::DeviceDiagnosticsConfigCreateInfoNV().setFlags(
        vk::DeviceDiagnosticsConfigFlagBitsNV::eEnableResourceTracking |
        vk::DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderDebugInfo |
        vk::DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderErrorReporting);

    auto shaderAtomicFloatFeatures =
        vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT()
            .setShaderBufferFloat32Atomics(true)
            .setShaderImageFloat32Atomics(true);

    pNext = nullptr;
    APPEND_EXTENSION(accelStructSupported, accelStructFeatures)
    APPEND_EXTENSION(rayPipelineSupported, rayPipelineFeatures)
    APPEND_EXTENSION(rayQuerySupported, rayQueryFeatures)
    APPEND_EXTENSION(meshletsSupported, meshletFeatures)
    APPEND_EXTENSION(vrsSupported, vrsFeatures)
    APPEND_EXTENSION(shaderAtomicSupported, shaderAtomicFloatFeatures)
    APPEND_EXTENSION(
        physicalDeviceProperties.apiVersion >= VK_API_VERSION_1_3,
        vulkan13features)
    APPEND_EXTENSION(
        physicalDeviceProperties.apiVersion < VK_API_VERSION_1_3 &&
            maintenance4Supported,
        maintenance4Features);
#if USTC_CG_WITH_AFTERMATH
    if (aftermathPhysicalFeatures.diagnosticsConfig &&
        m_DeviceParams.enableAftermath)
        APPEND_EXTENSION(aftermathSupported, aftermathFeatures);
#endif
#undef APPEND_EXTENSION

    auto deviceFeatures = vk::PhysicalDeviceFeatures()
                              .setShaderImageGatherExtended(true)
                              .setSamplerAnisotropy(true)
                              .setTessellationShader(true)
                              .setTextureCompressionBC(true)
                              .setGeometryShader(true)
                              .setImageCubeArray(true)
                              .setDualSrcBlend(true)
                              .setMultiDrawIndirect(true);

    auto vulkan12features =
        vk::PhysicalDeviceVulkan12Features()
            .setDescriptorIndexing(true)
            .setRuntimeDescriptorArray(true)
            .setDescriptorBindingPartiallyBound(true)
            .setDescriptorBindingVariableDescriptorCount(true)
            .setTimelineSemaphore(true)
            .setShaderSampledImageArrayNonUniformIndexing(true)
            .setBufferDeviceAddress(
                bufferDeviceAddressFeatures.bufferDeviceAddress)
            .setPNext(pNext);

    auto layerVec = stringSetToVector(enabledExtensions.layers);
    auto extVec = stringSetToVector(enabledExtensions.device);

    auto deviceDesc = vk::DeviceCreateInfo()
                          .setPQueueCreateInfos(queueDesc.data())
                          .setQueueCreateInfoCount(uint32_t(queueDesc.size()))
                          .setPEnabledFeatures(&deviceFeatures)
                          .setEnabledExtensionCount(uint32_t(extVec.size()))
                          .setPpEnabledExtensionNames(extVec.data())
                          .setEnabledLayerCount(uint32_t(layerVec.size()))
                          .setPpEnabledLayerNames(layerVec.data())
                          .setPNext(&vulkan12features);

    if (m_DeviceParams.deviceCreateInfoCallback)
        m_DeviceParams.deviceCreateInfoCallback(deviceDesc);

    const vk::Result res = m_VulkanPhysicalDevice.createDevice(
        &deviceDesc, nullptr, &m_VulkanDevice);
    if (res != vk::Result::eSuccess) {
        log::error(
            "Failed to create a Vulkan physical device, error code = %s",
            nvrhi::vulkan::resultToString(VkResult(res)));
        return false;
    }

    m_VulkanDevice.getQueue(m_GraphicsQueueFamily, 0, &m_GraphicsQueue);
    if (m_DeviceParams.enableComputeQueue)
        m_VulkanDevice.getQueue(m_ComputeQueueFamily, 0, &m_ComputeQueue);
    if (m_DeviceParams.enableCopyQueue)
        m_VulkanDevice.getQueue(m_TransferQueueFamily, 0, &m_TransferQueue);
    if (!m_DeviceParams.headlessDevice)
        m_VulkanDevice.getQueue(m_PresentQueueFamily, 0, &m_PresentQueue);

    VULKAN_HPP_DEFAULT_DISPATCHER.init(m_VulkanDevice);

    // remember the bufferDeviceAddress feature enablement
    m_BufferDeviceAddressSupported = vulkan12features.bufferDeviceAddress;

    log::message(
        m_DeviceParams.infoLogSeverity,
        "Created Vulkan device: %s",
        m_RendererString.c_str());

    return true;
}

bool DeviceManager_VK::createWindowSurface()
{
    const VkResult res = glfwCreateWindowSurface(
        m_VulkanInstance, m_Window, nullptr, (VkSurfaceKHR*)&m_WindowSurface);
    if (res != VK_SUCCESS) {
        log::error(
            "Failed to create a GLFW window surface, error code = %s",
            nvrhi::vulkan::resultToString(res));
        return false;
    }

    return true;
}

void DeviceManager_VK::destroySwapChain()
{
    if (m_VulkanDevice) {
        m_VulkanDevice.waitIdle();
    }

    if (m_SwapChain) {
        m_VulkanDevice.destroySwapchainKHR(m_SwapChain);
        m_SwapChain = nullptr;
    }

    m_SwapChainImages.clear();
}

bool DeviceManager_VK::createSwapChain()
{
    destroySwapChain();

    m_SwapChainFormat = { vk::Format(nvrhi::vulkan::convertFormat(
                              m_DeviceParams.swapChainFormat)),
                          vk::ColorSpaceKHR::eSrgbNonlinear };

    vk::Extent2D extent = vk::Extent2D(
        m_DeviceParams.backBufferWidth, m_DeviceParams.backBufferHeight);

    std::unordered_set<uint32_t> uniqueQueues = {
        uint32_t(m_GraphicsQueueFamily), uint32_t(m_PresentQueueFamily)
    };

    std::vector<uint32_t> queues = setToVector(uniqueQueues);

    const bool enableSwapChainSharing = queues.size() > 1;

    auto desc =
        vk::SwapchainCreateInfoKHR()
            .setSurface(m_WindowSurface)
            .setMinImageCount(m_DeviceParams.swapChainBufferCount)
            .setImageFormat(m_SwapChainFormat.format)
            .setImageColorSpace(m_SwapChainFormat.colorSpace)
            .setImageExtent(extent)
            .setImageArrayLayers(1)
            .setImageUsage(
                vk::ImageUsageFlagBits::eColorAttachment |
                vk::ImageUsageFlagBits::eTransferDst |
                vk::ImageUsageFlagBits::eSampled)
            .setImageSharingMode(
                enableSwapChainSharing ? vk::SharingMode::eConcurrent
                                       : vk::SharingMode::eExclusive)
            .setFlags(
                m_SwapChainMutableFormatSupported
                    ? vk::SwapchainCreateFlagBitsKHR::eMutableFormat
                    : vk::SwapchainCreateFlagBitsKHR(0))
            .setQueueFamilyIndexCount(
                enableSwapChainSharing ? uint32_t(queues.size()) : 0)
            .setPQueueFamilyIndices(
                enableSwapChainSharing ? queues.data() : nullptr)
            .setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity)
            .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
            .setPresentMode(
                m_DeviceParams.vsyncEnabled ? vk::PresentModeKHR::eFifo
                                            : vk::PresentModeKHR::eImmediate)
            .setClipped(true)
            .setOldSwapchain(nullptr);

    std::vector<vk::Format> imageFormats = { m_SwapChainFormat.format };
    switch (m_SwapChainFormat.format) {
        case vk::Format::eR8G8B8A8Unorm:
            imageFormats.push_back(vk::Format::eR8G8B8A8Srgb);
            break;
        case vk::Format::eR8G8B8A8Srgb:
            imageFormats.push_back(vk::Format::eR8G8B8A8Unorm);
            break;
        case vk::Format::eB8G8R8A8Unorm:
            imageFormats.push_back(vk::Format::eB8G8R8A8Srgb);
            break;
        case vk::Format::eB8G8R8A8Srgb:
            imageFormats.push_back(vk::Format::eB8G8R8A8Unorm);
            break;
        default: break;
    }

    auto imageFormatListCreateInfo =
        vk::ImageFormatListCreateInfo().setViewFormats(imageFormats);

    if (m_SwapChainMutableFormatSupported)
        desc.pNext = &imageFormatListCreateInfo;

    const vk::Result res =
        m_VulkanDevice.createSwapchainKHR(&desc, nullptr, &m_SwapChain);
    if (res != vk::Result::eSuccess) {
        log::error(
            "Failed to create a Vulkan swap chain, error code = %s",
            nvrhi::vulkan::resultToString(VkResult(res)));
        return false;
    }

    // retrieve swap chain images
    auto images = m_VulkanDevice.getSwapchainImagesKHR(m_SwapChain);
    for (auto image : images) {
        SwapChainImage sci;
        sci.image = image;

        nvrhi::TextureDesc textureDesc;
        textureDesc.width = m_DeviceParams.backBufferWidth;
        textureDesc.height = m_DeviceParams.backBufferHeight;
        textureDesc.format = m_DeviceParams.swapChainFormat;
        textureDesc.debugName = "Swap chain image";
        textureDesc.initialState = nvrhi::ResourceStates::Present;
        textureDesc.keepInitialState = true;
        textureDesc.isRenderTarget = true;

        sci.rhiHandle = m_NvrhiDevice->createHandleForNativeTexture(
            nvrhi::ObjectTypes::VK_Image,
            nvrhi::Object(sci.image),
            textureDesc);
        m_SwapChainImages.push_back(sci);
    }

    m_SwapChainIndex = 0;

    return true;
}

#define CHECK(a)      \
    if (!(a)) {       \
        return false; \
    }

bool DeviceManager_VK::CreateInstanceInternal()
{
    if (m_DeviceParams.enableDebugRuntime) {
        enabledExtensions.instance.insert("VK_EXT_debug_report");
        enabledExtensions.layers.insert("VK_LAYER_KHRONOS_validation");
    }
#if USTC_CG_WITH_AFTERMATH
    if (m_DeviceParams.enableAftermath) {
        enabledExtensions.instance.insert("VK_EXT_debug_utils");
    }
#endif

    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr =
        m_dynamicLoader.getProcAddress<PFN_vkGetInstanceProcAddr>(
            "vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    return createInstance();
}

bool DeviceManager_VK::EnumerateAdapters(std::vector<AdapterInfo>& outAdapters)
{
    if (!m_VulkanInstance)
        return false;

    std::vector<vk::PhysicalDevice> devices =
        m_VulkanInstance.enumeratePhysicalDevices();
    outAdapters.clear();

    for (auto physicalDevice : devices) {
        vk::PhysicalDeviceProperties properties =
            physicalDevice.getProperties();

        AdapterInfo adapterInfo;
        adapterInfo.name = properties.deviceName.data();
        adapterInfo.vendorID = properties.vendorID;
        adapterInfo.deviceID = properties.deviceID;
        adapterInfo.vkPhysicalDevice = physicalDevice;
        adapterInfo.dedicatedVideoMemory = 0;

        // Go through the memory types to figure out the amount of VRAM on this
        // physical device.
        vk::PhysicalDeviceMemoryProperties memoryProperties =
            physicalDevice.getMemoryProperties();
        for (uint32_t heapIndex = 0;
             heapIndex < memoryProperties.memoryHeapCount;
             ++heapIndex) {
            vk::MemoryHeap const& heap =
                memoryProperties.memoryHeaps[heapIndex];
            if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                adapterInfo.dedicatedVideoMemory += heap.size;
            }
        }

        outAdapters.push_back(std::move(adapterInfo));
    }

    return true;
}

bool DeviceManager_VK::CreateDevice()
{
    if (m_DeviceParams.enableDebugRuntime) {
        installDebugCallback();
    }

    // add device extensions requested by the user
    for (const std::string& name :
         m_DeviceParams.requiredVulkanDeviceExtensions) {
        enabledExtensions.device.insert(name);
    }
    for (const std::string& name :
         m_DeviceParams.optionalVulkanDeviceExtensions) {
        optionalExtensions.device.insert(name);
    }

    if (!m_DeviceParams.headlessDevice) {
        // Need to adjust the swap chain format before creating the device
        // because it affects physical device selection
        if (m_DeviceParams.swapChainFormat == nvrhi::Format::SRGBA8_UNORM)
            m_DeviceParams.swapChainFormat = nvrhi::Format::SBGRA8_UNORM;
        else if (m_DeviceParams.swapChainFormat == nvrhi::Format::RGBA8_UNORM)
            m_DeviceParams.swapChainFormat = nvrhi::Format::BGRA8_UNORM;

        CHECK(createWindowSurface())
    }
    CHECK(pickPhysicalDevice())
    CHECK(findQueueFamilies(m_VulkanPhysicalDevice))
    CHECK(createDevice())

    auto vecInstanceExt = stringSetToVector(enabledExtensions.instance);
    auto vecLayers = stringSetToVector(enabledExtensions.layers);
    auto vecDeviceExt = stringSetToVector(enabledExtensions.device);

    nvrhi::vulkan::DeviceDesc deviceDesc;
    deviceDesc.errorCB = &DefaultMessageCallback::GetInstance();
    deviceDesc.instance = m_VulkanInstance;
    deviceDesc.physicalDevice = m_VulkanPhysicalDevice;
    deviceDesc.device = m_VulkanDevice;
    deviceDesc.graphicsQueue = m_GraphicsQueue;
    deviceDesc.graphicsQueueIndex = m_GraphicsQueueFamily;
    if (m_DeviceParams.enableComputeQueue) {
        deviceDesc.computeQueue = m_ComputeQueue;
        deviceDesc.computeQueueIndex = m_ComputeQueueFamily;
    }
    if (m_DeviceParams.enableCopyQueue) {
        deviceDesc.transferQueue = m_TransferQueue;
        deviceDesc.transferQueueIndex = m_TransferQueueFamily;
    }
    deviceDesc.instanceExtensions = vecInstanceExt.data();
    deviceDesc.numInstanceExtensions = vecInstanceExt.size();
    deviceDesc.deviceExtensions = vecDeviceExt.data();
    deviceDesc.numDeviceExtensions = vecDeviceExt.size();
    deviceDesc.bufferDeviceAddressSupported = m_BufferDeviceAddressSupported;
#if USTC_CG_WITH_AFTERMATH
    deviceDesc.aftermathEnabled = m_DeviceParams.enableAftermath;
#endif

    m_NvrhiDevice = nvrhi::vulkan::createDevice(deviceDesc);

    if (m_DeviceParams.enableNvrhiValidationLayer) {
        m_ValidationLayer =
            nvrhi::validation::createValidationLayer(m_NvrhiDevice);
    }

    return true;
}

bool DeviceManager_VK::CreateSwapChain()
{
    CHECK(createSwapChain())

    m_BarrierCommandList = m_NvrhiDevice->createCommandList();

    m_PresentSemaphores.reserve(m_DeviceParams.maxFramesInFlight + 1);
    for (uint32_t i = 0; i < m_DeviceParams.maxFramesInFlight + 1; ++i) {
        m_PresentSemaphores.push_back(
            m_VulkanDevice.createSemaphore(vk::SemaphoreCreateInfo()));
    }

    return true;
}
#undef CHECK

void DeviceManager_VK::DestroyDeviceAndSwapChain()
{
    destroySwapChain();

    for (auto& semaphore : m_PresentSemaphores) {
        if (semaphore) {
            m_VulkanDevice.destroySemaphore(semaphore);
            semaphore = vk::Semaphore();
        }
    }

    m_BarrierCommandList = nullptr;

    m_NvrhiDevice = nullptr;
    m_ValidationLayer = nullptr;
    m_RendererString.clear();

    if (m_VulkanDevice) {
        m_VulkanDevice.destroy();
        m_VulkanDevice = nullptr;
    }

    if (m_WindowSurface) {
        assert(m_VulkanInstance);
        m_VulkanInstance.destroySurfaceKHR(m_WindowSurface);
        m_WindowSurface = nullptr;
    }

    if (m_DebugReportCallback) {
        m_VulkanInstance.destroyDebugReportCallbackEXT(m_DebugReportCallback);
    }

    if (m_VulkanInstance) {
        m_VulkanInstance.destroy();
        m_VulkanInstance = nullptr;
    }
}

bool DeviceManager_VK::BeginFrame()
{
    const auto& semaphore = m_PresentSemaphores[m_PresentSemaphoreIndex];

    vk::Result res;

    int const maxAttempts = 3;
    for (int attempt = 0; attempt < maxAttempts; ++attempt) {
        res = m_VulkanDevice.acquireNextImageKHR(
            m_SwapChain,
            std::numeric_limits<uint64_t>::max(),  // timeout
            semaphore,
            vk::Fence(),
            &m_SwapChainIndex);

        if (res == vk::Result::eErrorOutOfDateKHR && attempt < maxAttempts) {
            BackBufferResizing();
            auto surfaceCaps = m_VulkanPhysicalDevice.getSurfaceCapabilitiesKHR(
                m_WindowSurface);

            m_DeviceParams.backBufferWidth = surfaceCaps.currentExtent.width;
            m_DeviceParams.backBufferHeight = surfaceCaps.currentExtent.height;

            ResizeSwapChain();
            BackBufferResized();
        }
        else
            break;
    }

    if (res == vk::Result::eSuccess) {
        m_NvrhiDevice->queueWaitForSemaphore(
            nvrhi::CommandQueue::Graphics, semaphore, 0);
        return true;
    }

    return false;
}

bool DeviceManager_VK::Present()
{
    const auto& semaphore = m_PresentSemaphores[m_PresentSemaphoreIndex];

    m_NvrhiDevice->queueSignalSemaphore(
        nvrhi::CommandQueue::Graphics, semaphore, 0);

    m_BarrierCommandList->open();  // umm...
    m_BarrierCommandList->close();
    m_NvrhiDevice->executeCommandList(m_BarrierCommandList);

    vk::PresentInfoKHR info = vk::PresentInfoKHR()
                                  .setWaitSemaphoreCount(1)
                                  .setPWaitSemaphores(&semaphore)
                                  .setSwapchainCount(1)
                                  .setPSwapchains(&m_SwapChain)
                                  .setPImageIndices(&m_SwapChainIndex);

    const vk::Result res = m_PresentQueue.presentKHR(&info);
    if (!(res == vk::Result::eSuccess ||
          res == vk::Result::eErrorOutOfDateKHR)) {
        return false;
    }

    m_PresentSemaphoreIndex =
        (m_PresentSemaphoreIndex + 1) % m_PresentSemaphores.size();

#ifndef _WIN32
    if (m_DeviceParams.vsyncEnabled) {
        m_PresentQueue.waitIdle();
    }
#endif

    while (m_FramesInFlight.size() >= m_DeviceParams.maxFramesInFlight) {
        auto query = m_FramesInFlight.front();
        m_FramesInFlight.pop();

        m_NvrhiDevice->waitEventQuery(query);

        m_QueryPool.push_back(query);
    }

    nvrhi::EventQueryHandle query;
    if (!m_QueryPool.empty()) {
        query = m_QueryPool.back();
        m_QueryPool.pop_back();
    }
    else {
        query = m_NvrhiDevice->createEventQuery();
    }

    m_NvrhiDevice->resetEventQuery(query);
    m_NvrhiDevice->setEventQuery(query, nvrhi::CommandQueue::Graphics);
    m_FramesInFlight.push(query);
    return true;
}

DeviceManager* DeviceManager::CreateVK()
{
    return new DeviceManager_VK();
}

USTC_CG_NAMESPACE_CLOSE_SCOPE
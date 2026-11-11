// SPDX-FileCopyrightText: Copyright (c) 2026 The neraulGraphics Developers
// SPDX-License-Identifier: Apache-2.0
//
// Milestone V0 of the portable backend: build acceleration structures and trace rays through Vulkan's ray queries on whichever device
// is asked for, so that the result can be compared with the CPU reference on NVIDIA, AMD and a software device alike.
//
//   vk_rayquery rayquery.spv scene.bin hits.bin [device index | list]
//
// scene.bin: int32 instance count, int32 ray count, then 12 floats per instance (a 3x4 row-major affine matrix applied to a unit cube
// centred on the origin) and 6 floats per ray (origin, unit direction). hits.bin: per ray a float distance (-1 for a miss) and an int32
// instance index (-1 for a miss).
//
// Buffers that the GPU works on live in device-local memory and are filled through a staging buffer, as in a real backend; only the
// staging copies are host-visible.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#define VK(call)                                                                                                    \
    do {                                                                                                            \
        const VkResult r_ = (call);                                                                                 \
        if (r_ != VK_SUCCESS) {                                                                                     \
            std::fprintf(stderr, "%s:%d: %s failed with VkResult %d\n", __FILE__, __LINE__, #call, int(r_));        \
            std::exit(2);                                                                                           \
        }                                                                                                           \
    } while (0)

namespace {

VkDevice device;
VkPhysicalDevice physical;
VkPhysicalDeviceMemoryProperties memory_properties;

PFN_vkCreateAccelerationStructureKHR create_as;
PFN_vkDestroyAccelerationStructureKHR destroy_as;
PFN_vkGetAccelerationStructureBuildSizesKHR build_sizes;
PFN_vkCmdBuildAccelerationStructuresKHR cmd_build_as;
PFN_vkGetAccelerationStructureDeviceAddressKHR as_address;

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void *mapped = nullptr;
    VkDeviceAddress address = 0;
};

uint32_t find_memory(uint32_t type_bits, VkMemoryPropertyFlags wanted) {
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i)
        if ((type_bits & (1u << i)) && (memory_properties.memoryTypes[i].propertyFlags & wanted) == wanted) return i;
    std::fprintf(stderr, "no suitable memory type\n");
    std::exit(2);
}

Buffer make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, bool local = false) {
    Buffer b;
    b.size = std::max<VkDeviceSize>(size, 16);
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = b.size;
    info.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK(vkCreateBuffer(device, &info, nullptr, &b.buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, b.buffer, &req);
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = find_memory(req.memoryTypeBits, local ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                                                  : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    VK(vkAllocateMemory(device, &alloc, nullptr, &b.memory));
    VK(vkBindBufferMemory(device, b.buffer, b.memory, 0));
    if (!local) VK(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));
    VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    address.buffer = b.buffer;
    b.address = vkGetBufferDeviceAddress(device, &address);
    return b;
}

struct AccelerationStructure {
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    Buffer storage;
    VkDeviceAddress address = 0;
};

VkQueue queue;
VkCommandPool pool;
VkDeviceSize scratch_alignment = 128;

void destroy(Buffer &b) {
    vkDestroyBuffer(device, b.buffer, nullptr);
    vkFreeMemory(device, b.memory, nullptr);
}

// Runs a one-time command buffer and waits for it.
template <class Record>
void submit_now(Record record) {
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK(vkAllocateCommandBuffers(device, &allocate, &cmd));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK(vkBeginCommandBuffer(cmd, &begin));
    record(cmd);
    VK(vkEndCommandBuffer(cmd));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
    VK(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(device, pool, 1, &cmd);
}

void upload(const Buffer &dst, const void *data, VkDeviceSize bytes) {
    Buffer staging = make_buffer(bytes, 0);
    std::memcpy(staging.mapped, data, bytes);
    submit_now([&](VkCommandBuffer cmd) {
        const VkBufferCopy region{0, 0, bytes};
        vkCmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &region);
    });
    destroy(staging);
}

std::vector<char> download(const Buffer &src, VkDeviceSize bytes) {
    Buffer staging = make_buffer(bytes, 0);
    submit_now([&](VkCommandBuffer cmd) {
        const VkBufferCopy region{0, 0, bytes};
        vkCmdCopyBuffer(cmd, src.buffer, staging.buffer, 1, &region);
    });
    std::vector<char> out(bytes);
    std::memcpy(out.data(), staging.mapped, bytes);
    destroy(staging);
    return out;
}

// Builds one acceleration structure and waits for it.
AccelerationStructure build(VkAccelerationStructureTypeKHR type, const VkAccelerationStructureGeometryKHR &geometry, uint32_t primitives) {
    VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    info.type = type;
    info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info.geometryCount = 1;
    info.pGeometries = &geometry;
    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    build_sizes(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &primitives, &sizes);

    AccelerationStructure as;
    as.storage = make_buffer(sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, true);
    VkAccelerationStructureCreateInfoKHR create{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    create.buffer = as.storage.buffer;
    create.size = sizes.accelerationStructureSize;
    create.type = type;
    VK(create_as(device, &create, nullptr, &as.handle));

    Buffer scratch = make_buffer(sizes.buildScratchSize + scratch_alignment, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
    info.dstAccelerationStructure = as.handle;
    info.scratchData.deviceAddress = (scratch.address + scratch_alignment - 1) / scratch_alignment * scratch_alignment;

    VkAccelerationStructureBuildRangeInfoKHR range{primitives, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR *ranges = &range;
    submit_now([&](VkCommandBuffer cmd) { cmd_build_as(cmd, 1, &info, &ranges); });
    destroy(scratch);

    VkAccelerationStructureDeviceAddressInfoKHR address{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    address.accelerationStructure = as.handle;
    as.address = as_address(device, &address);
    return as;
}

std::vector<char> read_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        std::exit(1);
    }
    return std::vector<char>(std::istreambuf_iterator<char>(in), {});
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s rayquery.spv scene.bin hits.bin [device index | list]\n", argv[0]);
        return 1;
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    VkInstance instance;
    VK(vkCreateInstance(&instance_info, nullptr, &instance));

    uint32_t count = 0;
    VK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count);
    VK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
    if (argc > 4 && std::string(argv[4]) == "list") {
        for (uint32_t i = 0; i < count; ++i) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(devices[i], &p);
            std::printf("%u %s\n", i, p.deviceName);
        }
        return 0;
    }
    const uint32_t chosen = argc > 4 ? static_cast<uint32_t>(std::atoi(argv[4])) : 0;
    if (chosen >= count) {
        std::fprintf(stderr, "no device %u\n", chosen);
        return 1;
    }
    physical = devices[chosen];
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical, &properties);
    vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);

    VkPhysicalDeviceAccelerationStructurePropertiesKHR as_properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &as_properties};
    vkGetPhysicalDeviceProperties2(physical, &properties2);
    scratch_alignment = std::max<VkDeviceSize>(as_properties.minAccelerationStructureScratchOffsetAlignment, 1);

    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < family_count; ++i)
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { family = i; break; }
    if (family == UINT32_MAX) { std::fprintf(stderr, "no compute queue\n"); return 1; }

    VkPhysicalDeviceRayQueryFeaturesKHR ray_query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    ray_query.rayQuery = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR as_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &ray_query};
    as_features.accelerationStructure = VK_TRUE;
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &as_features};
    v12.bufferDeviceAddress = VK_TRUE;
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    const char *extensions[] = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &v12};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = 3;
    device_info.ppEnabledExtensionNames = extensions;
    VK(vkCreateDevice(physical, &device_info, nullptr, &device));
    vkGetDeviceQueue(device, family, 0, &queue);
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK(vkCreateCommandPool(device, &pool_info, nullptr, &pool));

    create_as = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR"));
    destroy_as = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR"));
    build_sizes = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR"));
    cmd_build_as = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));
    as_address = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR"));
    std::fprintf(stderr, "device %u: %s\n", chosen, properties.deviceName);

    // ---- the scene
    const std::vector<char> scene = read_file(argv[2]);
    const int32_t instance_count = reinterpret_cast<const int32_t *>(scene.data())[0];
    const int32_t ray_count = reinterpret_cast<const int32_t *>(scene.data())[1];
    const float *matrices = reinterpret_cast<const float *>(scene.data() + 8);
    const float *ray_data = matrices + 12 * instance_count;

    // a unit cube centred on the origin: 8 vertices, 12 triangles
    const float cube_vertices[8][3] = {{-.5f, -.5f, -.5f}, {.5f, -.5f, -.5f}, {.5f, .5f, -.5f}, {-.5f, .5f, -.5f},
                                       {-.5f, -.5f, .5f},  {.5f, -.5f, .5f},  {.5f, .5f, .5f},  {-.5f, .5f, .5f}};
    const uint32_t cube_indices[36] = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3};
    const VkBufferUsageFlags build_input = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    Buffer vertex_buffer = make_buffer(sizeof(cube_vertices), build_input | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
    Buffer index_buffer = make_buffer(sizeof(cube_indices), build_input | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
    upload(vertex_buffer, cube_vertices, sizeof(cube_vertices));
    upload(index_buffer, cube_indices, sizeof(cube_indices));

    VkAccelerationStructureGeometryKHR blas_geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    blas_geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    blas_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    blas_geometry.geometry.triangles = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    blas_geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    blas_geometry.geometry.triangles.vertexData.deviceAddress = vertex_buffer.address;
    blas_geometry.geometry.triangles.vertexStride = 3 * sizeof(float);
    blas_geometry.geometry.triangles.maxVertex = 7;
    blas_geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    blas_geometry.geometry.triangles.indexData.deviceAddress = index_buffer.address;
    AccelerationStructure blas = build(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, blas_geometry, 12);

    std::vector<VkAccelerationStructureInstanceKHR> instances(instance_count);
    for (int32_t i = 0; i < instance_count; ++i) {
        VkAccelerationStructureInstanceKHR &inst = instances[i];
        std::memset(&inst, 0, sizeof(inst));
        std::memcpy(&inst.transform, matrices + 12 * i, 12 * sizeof(float));
        inst.instanceCustomIndex = static_cast<uint32_t>(i);
        inst.mask = 0xFF;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = blas.address;
    }
    Buffer instance_buffer = make_buffer(sizeof(VkAccelerationStructureInstanceKHR) * std::max(instance_count, 1), build_input | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
    upload(instance_buffer, instances.data(), sizeof(VkAccelerationStructureInstanceKHR) * instance_count);
    VkAccelerationStructureGeometryKHR tlas_geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    tlas_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlas_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    tlas_geometry.geometry.instances = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    tlas_geometry.geometry.instances.data.deviceAddress = instance_buffer.address;
    AccelerationStructure tlas = build(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, tlas_geometry, static_cast<uint32_t>(instance_count));

    // ---- the compute pipeline
    Buffer ray_buffer = make_buffer(sizeof(float) * 6 * std::max(ray_count, 1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
    Buffer hit_buffer = make_buffer(sizeof(uint32_t) * 2 * std::max(ray_count, 1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
    upload(ray_buffer, ray_data, sizeof(float) * 6 * ray_count);

    const std::vector<char> spirv = read_file(argv[1]);
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = spirv.size();
    module_info.pCode = reinterpret_cast<const uint32_t *>(spirv.data());
    VkShaderModule module;
    VK(vkCreateShaderModule(device, &module_info, nullptr, &module));

    VkDescriptorSetLayoutBinding bindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = 3;
    layout_info.pBindings = bindings;
    VkDescriptorSetLayout set_layout;
    VK(vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &set_layout));
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push;
    VkPipelineLayout pipeline_layout;
    VK(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout));
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
    pipeline_info.layout = pipeline_layout;
    VkPipeline pipeline;
    VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline));

    VkDescriptorPoolSize pool_sizes[2] = {{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}};
    VkDescriptorPoolCreateInfo descriptor_pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    descriptor_pool_info.maxSets = 1;
    descriptor_pool_info.poolSizeCount = 2;
    descriptor_pool_info.pPoolSizes = pool_sizes;
    VkDescriptorPool descriptor_pool;
    VK(vkCreateDescriptorPool(device, &descriptor_pool_info, nullptr, &descriptor_pool));
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = descriptor_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &set_layout;
    VkDescriptorSet set;
    VK(vkAllocateDescriptorSets(device, &set_info, &set));
    VkWriteDescriptorSetAccelerationStructureKHR write_as{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    write_as.accelerationStructureCount = 1;
    write_as.pAccelerationStructures = &tlas.handle;
    VkDescriptorBufferInfo ray_info{ray_buffer.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo hit_info{hit_buffer.buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet writes[3] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &write_as, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr},
                                      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &ray_info, nullptr},
                                      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &hit_info, nullptr}};
    vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK(vkAllocateCommandBuffers(device, &allocate, &cmd));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK(vkBeginCommandBuffer(cmd, &begin));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, nullptr);
    const uint32_t n = static_cast<uint32_t>(ray_count);
    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(n), &n);
    vkCmdDispatch(cmd, (n + 63) / 64, 1, 1);
    VK(vkEndCommandBuffer(cmd));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    VK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));     // a first run warms the device up
    VK(vkQueueWaitIdle(queue));
    const int repeats = 5;
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) {
        VK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
        VK(vkQueueWaitIdle(queue));
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / repeats;
    std::fprintf(stderr, "%d rays against %d instances: %.3f ms (%.1f Mrays/s)\n", ray_count, instance_count, ms, ray_count / ms / 1000.0);

    std::ofstream out(argv[3], std::ios::binary);
    const std::vector<char> hit_bytes = download(hit_buffer, sizeof(uint32_t) * 2 * ray_count);
    const uint32_t *hits = reinterpret_cast<const uint32_t *>(hit_bytes.data());
    for (int32_t i = 0; i < ray_count; ++i) {
        float t;
        std::memcpy(&t, &hits[2 * i], 4);
        const int32_t id = hits[2 * i + 1] == 0xFFFFFFFFu ? -1 : static_cast<int32_t>(hits[2 * i + 1]);
        out.write(reinterpret_cast<const char *>(&t), 4);
        out.write(reinterpret_cast<const char *>(&id), 4);
    }
    return 0;
}

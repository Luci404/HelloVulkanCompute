#include <iostream>
#include <cassert>
#include <fstream>

#include <vulkan/vulkan.hpp>

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

int main(int argc, char** argv)
{
	// Initialize instance and application
	vk::ApplicationInfo applicationInfo{ "HelloVulkanHpp", 1, nullptr, 0, VK_API_VERSION_1_1 };
	const std::vector<const char*> layers = { "VK_LAYER_KHRONOS_validation" };
	vk::InstanceCreateInfo instanceCreateInfo(vk::InstanceCreateFlags(), &applicationInfo, layers, {});
	vk::Instance instance = vk::createInstance(instanceCreateInfo);

	// Find physical device.
	vk::PhysicalDevice physicalDevice = instance.enumeratePhysicalDevices().front();
	vk::PhysicalDeviceProperties physicalDeviceProperties = physicalDevice.getProperties();
	std::cout << "Device name: " << physicalDeviceProperties.deviceName << std::endl;
	std::cout << "Vulkan version: " << VK_VERSION_MAJOR(physicalDeviceProperties.apiVersion) << "."
		<< VK_VERSION_MINOR(physicalDeviceProperties.apiVersion) << "."
		<< VK_VERSION_PATCH(physicalDeviceProperties.apiVersion) << std::endl;
	std::cout << "Max compute shared memory size: " << physicalDeviceProperties.limits.maxComputeSharedMemorySize << std::endl;

	// Find compute queue family index.
	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
	auto computeQueueFamilyIterator = std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
		[](const vk::QueueFamilyProperties& p) { return p.queueFlags & vk::QueueFlagBits::eCompute; });
	const uint32_t computeQueueFamilyIndex = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), computeQueueFamilyIterator));
	std::cout << "Compute Queue Family Index: " << computeQueueFamilyIndex << std::endl;

	// Create logical device.
	const float queuePriority = 1.0f;
	vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), computeQueueFamilyIndex, 1, &queuePriority);
	vk::DeviceCreateInfo deviceCreateInfo(vk::DeviceCreateFlags(), deviceQueueCreateInfo);
	vk::Device device = physicalDevice.createDevice(deviceCreateInfo);

	// Allocate memory (aka. create buffers).
	const uint32_t numElements = 10;
	const uint32_t bufferSize = numElements * sizeof(int32_t);

	vk::BufferCreateInfo bufferCreateInfo(vk::BufferCreateFlags(), bufferSize, vk::BufferUsageFlagBits::eStorageBuffer, vk::SharingMode::eExclusive, 1, &computeQueueFamilyIndex);
	
	VmaAllocatorCreateInfo allocatorInfo = {};
	allocatorInfo.vulkanApiVersion = physicalDeviceProperties.apiVersion;
	allocatorInfo.physicalDevice = physicalDevice;
	allocatorInfo.device = device;
	allocatorInfo.instance = instance;

	VmaAllocator allocator;
	vmaCreateAllocator(&allocatorInfo, &allocator);

	VkBuffer inBufferRaw;
	VkBuffer outBufferRaw;

	VmaAllocationCreateInfo allocationInfo = {};
	allocationInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

	VmaAllocation inBufferAllocation;
	vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferCreateInfo), &allocationInfo, &inBufferRaw, &inBufferAllocation, nullptr);

	allocationInfo.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
	VmaAllocation outBufferAllocation;
	vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferCreateInfo), &allocationInfo, &outBufferRaw, &outBufferAllocation, nullptr);

	vk::Buffer inBuffer = inBufferRaw;
	vk::Buffer outBuffer = outBufferRaw;

	int32_t* inBufferPtr = nullptr;
	vmaMapMemory(allocator, inBufferAllocation, reinterpret_cast<void**>(&inBufferPtr));
	for (int32_t i = 0; i < numElements; ++i) inBufferPtr[i] = i;
	vmaUnmapMemory(allocator, inBufferAllocation);
	
	/*vk::Buffer inBuffer = device.createBuffer(bufferCreateInfo);
	vk::Buffer outBuffer = device.createBuffer(bufferCreateInfo);

	vk::MemoryRequirements inBufferMemoryRequirements = device.getBufferMemoryRequirements(inBuffer);
	vk::MemoryRequirements outBufferMemoryRequirements = device.getBufferMemoryRequirements(outBuffer);

	vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();

	uint32_t memoryTypeIndex = uint32_t(~0);
	vk::DeviceSize memoryHeapSize = uint32_t(~0);
	for (uint32_t currentMemoryTypeIndex = 0; currentMemoryTypeIndex < memoryProperties.memoryTypeCount; ++currentMemoryTypeIndex)
	{
		vk::MemoryType memoryType = memoryProperties.memoryTypes[currentMemoryTypeIndex];
		if ((vk::MemoryPropertyFlagBits::eHostVisible & memoryType.propertyFlags) &&
			(vk::MemoryPropertyFlagBits::eHostCoherent & memoryType.propertyFlags))
		{
			memoryHeapSize = memoryProperties.memoryHeaps[memoryType.heapIndex].size;
			memoryTypeIndex = currentMemoryTypeIndex;
			break;
		}
	}

	std::cout << "Memory Type Index: " << memoryTypeIndex << std::endl;
	std::cout << "Memory Heap Size : " << memoryHeapSize / 1024 / 1024 / 1024 << " GB" << std::endl;

	vk::MemoryAllocateInfo inBufferMemoryAllocateInfo(inBufferMemoryRequirements.size, memoryTypeIndex);
	vk::MemoryAllocateInfo outBufferMemoryAllocateInfo(outBufferMemoryRequirements.size, memoryTypeIndex);
	vk::DeviceMemory inBufferMemory = device.allocateMemory(inBufferMemoryAllocateInfo);
	vk::DeviceMemory outBufferMemory = device.allocateMemory(inBufferMemoryAllocateInfo);

	// Get a mapped pointer to the memory that can be used to copy data from the host to the device. Fill the inBuffer.
	int32_t* inBufferPtr = static_cast<int32_t*>(device.mapMemory(inBufferMemory, 0, bufferSize));
	for (int32_t i = 0; i < numElements; ++i) inBufferPtr[i] = i;
	device.unmapMemory(inBufferMemory);

	device.bindBufferMemory(inBuffer, inBufferMemory, 0);
	device.bindBufferMemory(outBuffer, outBufferMemory, 0);*/

	// Create compute pipeline.
	std::vector<char> shaderContents;
	if (std::ifstream shaderFile{ "D:/Dev/Luci404/HelloVulkanHpp/square.spv", std::ios::binary | std::ios::ate })
	{
		const size_t fileSize = shaderFile.tellg();
		shaderFile.seekg(0);
		shaderContents.resize(fileSize, '\0');
		shaderFile.read(shaderContents.data(), fileSize);
	}

	vk::ShaderModuleCreateInfo shaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), shaderContents.size(), reinterpret_cast<const uint32_t*>(shaderContents.data()));
	vk::ShaderModule shaderModule = device.createShaderModule(shaderModuleCreateInfo);

	// Descriptor set layout.
	const std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBinding = {
		{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
		{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute} };
	vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), descriptorSetLayoutBinding);
	vk::DescriptorSetLayout descriptorSetLayout = device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);

	// Pipeline layout.
	vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), descriptorSetLayout);
	vk::PipelineLayout pipelineLayout = device.createPipelineLayout(pipelineLayoutCreateInfo);
	vk::PipelineCache pipelineCache = device.createPipelineCache(vk::PipelineCacheCreateInfo());

	// Piepline.
	vk::PipelineShaderStageCreateInfo pipelineShaderCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, shaderModule, "Main");
	vk::ComputePipelineCreateInfo computePipelineCreateInfo(vk::PipelineCreateFlags(), pipelineShaderCreateInfo, pipelineLayout);
	vk::Pipeline computePipeline = device.createComputePipeline(pipelineCache, computePipelineCreateInfo).value;

	// Descriptor set.
	vk::DescriptorPoolSize descriptorPoolSize(vk::DescriptorType::eStorageBuffer, 2);
	vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), 1, descriptorPoolSize);
	vk::DescriptorPool descriptorPool = device.createDescriptorPool(descriptorPoolCreateInfo);

	vk::DescriptorSetAllocateInfo descriptorSetAllocInfo(descriptorPool, 1, &descriptorSetLayout);
	const std::vector<vk::DescriptorSet> descriptorSets = device.allocateDescriptorSets(descriptorSetAllocInfo);
	vk::DescriptorSet descriptorSet = descriptorSets.front();
	vk::DescriptorBufferInfo inBufferInfo(inBuffer, 0, numElements * sizeof(int32_t));
	vk::DescriptorBufferInfo outBufferInfo(outBuffer, 0, numElements * sizeof(int32_t));

	const std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
		{ descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &inBufferInfo},
		{ descriptorSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &outBufferInfo} };
	device.updateDescriptorSets(writeDescriptorSets, {});

	// Submit to GPU.
	vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlags(), computeQueueFamilyIndex);
	vk::CommandPool commandPool = device.createCommandPool(commandPoolCreateInfo);

	vk::CommandBufferAllocateInfo commandBufferAllocInfo(commandPool, vk::CommandBufferLevel::ePrimary, 1);
	const std::vector<vk::CommandBuffer> commandBuffers = device.allocateCommandBuffers(commandBufferAllocInfo);
	vk::CommandBuffer commandBuffer = commandBuffers.front();

	// Record commands.
	vk::CommandBufferBeginInfo cmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
	commandBuffer.begin(cmdBufferBeginInfo);
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, { descriptorSet }, {});
	commandBuffer.dispatch(numElements, 1, 1);
	commandBuffer.end();

	vk::Queue queue = device.getQueue(computeQueueFamilyIndex, 0);
	vk::Fence fence = device.createFence(vk::FenceCreateInfo());

	vk::SubmitInfo SubmitInfo(0, nullptr, nullptr, 1, &commandBuffer);
	queue.submit({ SubmitInfo }, fence);
	vk::Result result = device.waitForFences({ fence }, true, uint64_t(-1));
	assert(result == vk::Result::eSuccess);

	// Read results.
	vmaMapMemory(allocator, inBufferAllocation, reinterpret_cast<void**>(&inBufferPtr));
	for (uint32_t i = 0; i < numElements; ++i) std::cout << inBufferPtr[i] << " ";
	std::cout << std::endl;
	vmaUnmapMemory(allocator, inBufferAllocation);

	int32_t* outBufferPtr = nullptr;
	vmaMapMemory(allocator, outBufferAllocation, reinterpret_cast<void**>(&outBufferPtr));
	for (uint32_t i = 0; i < numElements; ++i) std::cout << outBufferPtr[i] << " ";
	std::cout << std::endl;
	vmaUnmapMemory(allocator, outBufferAllocation);

	struct BufferInfo
	{
		VkBuffer buffer;
		VmaAllocation allocation;
	};

	// Lets allocate a couple of buffers to see how they are layed out in memory
	auto AllocateBuffer = [allocator, computeQueueFamilyIndex](size_t SizeInBytes, VmaMemoryUsage Usage)
	{
		vk::BufferCreateInfo bufferCreateInfo{
			vk::BufferCreateFlags(),					// Flags
			SizeInBytes,								// Size
			vk::BufferUsageFlagBits::eStorageBuffer,	// Usage
			vk::SharingMode::eExclusive,				// Sharing mode
			1,											// Number of queue family indices
			&computeQueueFamilyIndex					// List of queue family indices
		};

		VmaAllocationCreateInfo allocationInfo = {};
		allocationInfo.usage = Usage;

		BufferInfo info;
		vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferCreateInfo), &allocationInfo, &info.buffer, &info.allocation, nullptr);

		return info;
	};

	auto DestroyBuffer = [allocator](BufferInfo info)
	{
		vmaDestroyBuffer(allocator, info.buffer, info.allocation);
	};

	constexpr size_t MB = 1024 * 1024;
	BufferInfo B1 = AllocateBuffer(4 * MB, VMA_MEMORY_USAGE_CPU_TO_GPU);
	BufferInfo B2 = AllocateBuffer(10 * MB, VMA_MEMORY_USAGE_GPU_TO_CPU);
	BufferInfo B3 = AllocateBuffer(20 * MB, VMA_MEMORY_USAGE_GPU_ONLY);
	BufferInfo B4 = AllocateBuffer(100 * MB, VMA_MEMORY_USAGE_CPU_ONLY);

	{
		char* statisticsString = nullptr;
		vmaBuildStatsString(allocator, &statisticsString, true);
		{
			std::ofstream file{ "VMAStatistics_2.json" };
			file << statisticsString;
		}
		vmaFreeStatsString(allocator, statisticsString);
	}

	DestroyBuffer(B1);
	DestroyBuffer(B2);
	DestroyBuffer(B3);
	DestroyBuffer(B4);
	/*inBufferPtr = static_cast<int32_t*>(device.mapMemory(inBufferMemory, 0, bufferSize));
	for (uint32_t i = 0; i < numElements; ++i) std::cout << inBufferPtr[i] << " ";
	std::cout << std::endl;
	device.unmapMemory(inBufferMemory);

	int32_t* outBufferPtr = static_cast<int32_t*>(device.mapMemory(outBufferMemory, 0, bufferSize));
	for (uint32_t i = 0; i < numElements; ++i) std::cout << outBufferPtr[i] << " ";
	std::cout << std::endl;
	device.unmapMemory(outBufferMemory);*/

	{
		char* statisticsString = nullptr;
		vmaBuildStatsString(allocator, &statisticsString, true);
		{
			std::ofstream file{ "VMAStatistics.json" };
			file << statisticsString;
		}
		vmaFreeStatsString(allocator, statisticsString);
	}

	vmaDestroyBuffer(allocator, inBuffer, inBufferAllocation);
	vmaDestroyBuffer(allocator, outBuffer, outBufferAllocation);
	vmaDestroyAllocator(allocator);

	// Cleaning.
	device.resetCommandPool(commandPool, vk::CommandPoolResetFlags());
	device.destroyFence(fence);
	device.destroyDescriptorSetLayout(descriptorSetLayout);
	device.destroyPipelineLayout(pipelineLayout);
	device.destroyPipelineCache(pipelineCache);
	device.destroyShaderModule(shaderModule);
	device.destroyPipeline(computePipeline);
	device.destroyDescriptorPool(descriptorPool);
	device.destroyCommandPool(commandPool);
	/*device.freeMemory(inBufferMemory);
	device.freeMemory(outBufferMemory);
	device.destroyBuffer(inBuffer);
	device.destroyBuffer(outBuffer);*/
	device.destroy();
	instance.destroy();

	return 0;
}
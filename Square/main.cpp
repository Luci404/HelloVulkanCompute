#include <iostream>
#include <cassert>
#include <fstream>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

int main(int argc, char** argv)
{
	// Create instance.
	VkApplicationInfo applicationInfo{
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pNext = nullptr,
		.pApplicationName = "Square",
		.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
		.pEngineName = "None",
		.engineVersion = VK_MAKE_VERSION(1, 0, 0),
		.apiVersion = VK_API_VERSION_1_1
	};

	VkInstanceCreateInfo instanceCreateInfo{
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.pApplicationInfo = &applicationInfo,
		.enabledLayerCount = 0,
		.ppEnabledLayerNames = nullptr,
		.enabledExtensionCount = 0,
		.ppEnabledExtensionNames = nullptr
	};

	VkInstance instance;
	assert(vkCreateInstance(&instanceCreateInfo, nullptr, &instance) == VK_SUCCESS);

	// Physical device.
	uint32_t physicalDeviceCount = 0;
	vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);
	assert(physicalDeviceCount > 0 && "Failed to find GPUs with Vulkan support!");
	std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
	vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data());
	VkPhysicalDevice physicalDevice = physicalDevices.front();

	VkPhysicalDeviceProperties physicalDeviceProperties;
	vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
	std::cout << "Device name: " << physicalDeviceProperties.deviceName << std::endl;
	std::cout << "Vulkan version: "
		<< VK_VERSION_MAJOR(physicalDeviceProperties.apiVersion) << "."
		<< VK_VERSION_MINOR(physicalDeviceProperties.apiVersion) << "."
		<< VK_VERSION_PATCH(physicalDeviceProperties.apiVersion) << std::endl;
	std::cout << "Max compute shared memory size: " << physicalDeviceProperties.limits.maxComputeSharedMemorySize << std::endl;

	
	// Compute queue family index.
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());

	auto computeQueueFamilyIterator = std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
		[](const VkQueueFamilyProperties& p) { return p.queueFlags & VK_QUEUE_COMPUTE_BIT; });
	const uint32_t computeQueueFamilyIndex = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), computeQueueFamilyIterator));
	std::cout << "Compute Queue Family Index: " << computeQueueFamilyIndex << std::endl;

	// Create logical device.
	const std::vector<float> queuePriorities = { 1.0f };
	VkDeviceQueueCreateInfo deviceQueueCreateInfo{
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO ,
		.pNext = nullptr,
		.flags = 0,
		.queueFamilyIndex = computeQueueFamilyIndex,
		.queueCount = 1,
		.pQueuePriorities = queuePriorities.data()
	};

	VkPhysicalDeviceFeatures physicalDeviceFeatures{};

	VkDeviceCreateInfo deviceCreateInfo{
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &deviceQueueCreateInfo,
		.enabledLayerCount = 0,
		.ppEnabledLayerNames = nullptr,
		.enabledExtensionCount = 0,
		.ppEnabledExtensionNames = nullptr,
		.pEnabledFeatures = &physicalDeviceFeatures
	};

	VkDevice logicalDevice;
	assert(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &logicalDevice) == VK_SUCCESS);

	// Allocate buffers.
	const uint32_t numElements = 10;
	const uint32_t bufferSize = numElements * sizeof(int32_t);

	VkBufferCreateInfo bufferCreateInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.size = bufferSize,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = &computeQueueFamilyIndex
	};

	VmaAllocatorCreateInfo allocatorCreateInfo{
		.physicalDevice = physicalDevice,
		.device = logicalDevice,
		.instance = instance,
		.vulkanApiVersion = physicalDeviceProperties.apiVersion
	};

	VmaAllocator allocator;
	vmaCreateAllocator(&allocatorCreateInfo, &allocator);

	VkBuffer inBufferRaw;
	VkBuffer outBufferRaw;

	VmaAllocationCreateInfo allocationInfo = {};
	allocationInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

	VmaAllocation inBufferAllocation;
	vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferCreateInfo), &allocationInfo, &inBufferRaw, &inBufferAllocation, nullptr);

	allocationInfo.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
	VmaAllocation outBufferAllocation;
	vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferCreateInfo), &allocationInfo, &outBufferRaw, &outBufferAllocation, nullptr);

	VkBuffer inBuffer = inBufferRaw;
	VkBuffer outBuffer = outBufferRaw;

	int32_t* inBufferPtr = nullptr;
	vmaMapMemory(allocator, inBufferAllocation, reinterpret_cast<void**>(&inBufferPtr));
	for (int32_t i = 0; i < numElements; ++i) inBufferPtr[i] = i;
	vmaUnmapMemory(allocator, inBufferAllocation);

	// Create compute pipeline.
	std::vector<char> shaderContents;
	if (std::ifstream shaderFile{ "D:/Dev/Luci404/HelloVulkanHpp/square.spv", std::ios::binary | std::ios::ate })
	{
		const size_t fileSize = shaderFile.tellg();
		shaderFile.seekg(0);
		shaderContents.resize(fileSize, '\0');
		shaderFile.read(shaderContents.data(), fileSize);
	}

	VkShaderModuleCreateInfo shaderModuleCreateInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.codeSize = shaderContents.size(),
		.pCode = reinterpret_cast<const uint32_t*>(shaderContents.data()),
	};

	const std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBinding = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
		},
	};

	// Cleanup.
	vkDestroyDevice(logicalDevice, nullptr);
	vkDestroyInstance(instance, nullptr);

	/*
	
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

	device.destroy();
	instance.destroy();*/

	return 0;
}
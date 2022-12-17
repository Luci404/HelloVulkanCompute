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

	VkShaderModule shaderModule;
	assert(vkCreateShaderModule(logicalDevice, &shaderModuleCreateInfo, nullptr, &shaderModule) == VK_SUCCESS);

	// Descriptor set layout.
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

	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.bindingCount = static_cast<uint32_t>(descriptorSetLayoutBinding.size()),
		.pBindings = descriptorSetLayoutBinding.data()
	};

	VkDescriptorSetLayout descriptorSetLayout;
	assert(vkCreateDescriptorSetLayout(logicalDevice, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout) == VK_SUCCESS);

	// Pipeline layout.
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptorSetLayout,
		.pushConstantRangeCount = 0,
		.pPushConstantRanges = nullptr,
	};

	VkPipelineLayout pipelineLayout;
	assert(vkCreatePipelineLayout(logicalDevice, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) == VK_SUCCESS);

	VkPipelineCacheCreateInfo pipelineCacheCreateInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.initialDataSize = 0,
		.pInitialData = nullptr
	};

	VkPipelineCache pipelineCache;
	assert(vkCreatePipelineCache(logicalDevice, &pipelineCacheCreateInfo, nullptr, &pipelineCache) == VK_SUCCESS);

	// Piepline.
	VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.stage = VK_SHADER_STAGE_COMPUTE_BIT,
		.module = shaderModule,
		.pName = "Main",
		.pSpecializationInfo = VK_NULL_HANDLE,
	};

	VkComputePipelineCreateInfo computePipelineCreateInfo{
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.stage = pipelineShaderStageCreateInfo,
		.layout = pipelineLayout,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = 0,
	};

	VkPipeline computePipeline;
	assert(vkCreateComputePipelines(logicalDevice, pipelineCache, 0, &computePipelineCreateInfo, nullptr, &computePipeline) == VK_SUCCESS);

	// Descriptor set.
	VkDescriptorPoolSize descriptorPoolSize{
		.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 2,
	};

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.maxSets = 1,
		.poolSizeCount = 1,
		.pPoolSizes = &descriptorPoolSize,
	};

	VkDescriptorPool descriptorPool;
	assert(vkCreateDescriptorPool(logicalDevice, &descriptorPoolCreateInfo, nullptr, &descriptorPool) == VK_SUCCESS);

	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.pNext = nullptr,
		.descriptorPool = descriptorPool,
		.descriptorSetCount = 1,
		.pSetLayouts = &descriptorSetLayout,
	};

	std::vector<VkDescriptorSet> descriptorSets(descriptorSetAllocateInfo.descriptorSetCount);
	assert(vkAllocateDescriptorSets(logicalDevice, &descriptorSetAllocateInfo, &descriptorSets[0]) == VK_SUCCESS);
	VkDescriptorSet descriptorSet = descriptorSets.front();


	VkDescriptorBufferInfo inputDescriptorBufferInfo{
		.buffer = inBuffer,
		.offset = 0,
		.range = numElements * sizeof(int32_t)
	};

	VkDescriptorBufferInfo outputDescriptorBufferInfo{
		.buffer = outBuffer,
		.offset = 0,
		.range = numElements * sizeof(int32_t)
	};

	std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.pNext = nullptr,
			.dstSet = descriptorSet,
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pImageInfo = nullptr,
			.pBufferInfo = &inputDescriptorBufferInfo,
			.pTexelBufferView = nullptr,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.pNext = nullptr,
			.dstSet = descriptorSet,
			.dstBinding = 1,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pImageInfo = nullptr,
			.pBufferInfo = &outputDescriptorBufferInfo,
			.pTexelBufferView = nullptr,
		}
	};

	vkUpdateDescriptorSets(logicalDevice, 2, &writeDescriptorSets[0], 0, nullptr);

	// Submit to GPU.
	VkCommandPoolCreateInfo commandPoolCreateInfo{
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.queueFamilyIndex = computeQueueFamilyIndex
	};

	VkCommandPool commandPool;
	assert(vkCreateCommandPool(logicalDevice, &commandPoolCreateInfo, nullptr, &commandPool) == VK_SUCCESS);


	VkCommandBufferAllocateInfo commandBufferAllocateInfo{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = nullptr,
		.commandPool = commandPool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};

	VkCommandBuffer commandBuffer;
	assert(vkAllocateCommandBuffers(logicalDevice, &commandBufferAllocateInfo, &commandBuffer) == VK_SUCCESS);

	// Record commands.
	VkCommandBufferBeginInfo commandBufferBeginInfo{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = nullptr,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		.pInheritanceInfo = nullptr,
	};
	vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
	vkCmdDispatch(commandBuffer, numElements, 1, 1);
	vkEndCommandBuffer(commandBuffer);

	VkQueue queue;
	vkGetDeviceQueue(logicalDevice, computeQueueFamilyIndex, 0, &queue);

	VkFenceCreateInfo fenceCreateInfo{
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
	};
	VkFence fence;
	vkCreateFence(logicalDevice, &fenceCreateInfo, nullptr, &fence);

	VkSubmitInfo submitInfo{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = nullptr,
		.waitSemaphoreCount = 0,
		.pWaitSemaphores = nullptr,
		.pWaitDstStageMask = nullptr,
		.commandBufferCount = 1,
		.pCommandBuffers = &commandBuffer,
		.signalSemaphoreCount = 0,
		.pSignalSemaphores = nullptr,
	};

	vkQueueSubmit(queue, 0, &submitInfo, fence);
	assert(vkWaitForFences(logicalDevice, 1, &fence, true, -1) == VK_SUCCESS);

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

	// Cleanup.
	vkResetCommandPool(logicalDevice, commandPool, 0);
	vkDestroyFence(logicalDevice, fence, nullptr);
	vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
	vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
	vkDestroyPipelineCache(logicalDevice, pipelineCache, nullptr);
	vkDestroyShaderModule(logicalDevice, shaderModule, nullptr);
	vkDestroyPipeline(logicalDevice, computePipeline, nullptr);
	vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
	vkDestroyCommandPool(logicalDevice, commandPool, nullptr);

	vkDestroyDevice(logicalDevice, nullptr);
	vkDestroyInstance(instance, nullptr);

	return 0;
}
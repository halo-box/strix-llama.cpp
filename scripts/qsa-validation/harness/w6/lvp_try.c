// compile + dispatch one SPIR-V on the first device: lvp_try file.spv n_bindings [spec constants...]
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define C(x) do { VkResult r_ = (x); if (r_) { printf("vk error %d line %d\n", r_, __LINE__); exit(1);} } while (0)
int main(int ac, char **av) {
    FILE *f = fopen(av[1], "rb"); fseek(f,0,SEEK_END); long n = ftell(f); fseek(f,0,SEEK_SET); uint32_t *code = malloc(n); fread(code,1,n,f); fclose(f);
    int nb = atoi(av[2]); int ns = ac - 3; uint32_t spec[16]; VkSpecializationMapEntry me[16];
    for (int i=0;i<ns;i++){spec[i]=atoi(av[3+i]); me[i].constantID=i; me[i].offset=4*i; me[i].size=4;}
    VkInstance inst; VkApplicationInfo ai={VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&ai}; C(vkCreateInstance(&ici,0,&inst));
    uint32_t c=1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst,&c,&pd); VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd,&pp); printf("device %s\n",pp.deviceName);
    float prio=1; VkDeviceQueueCreateInfo qi={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=0,.queueCount=1,.pQueuePriorities=&prio};
    VkPhysicalDeviceVulkan12Features f12={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.shaderFloat16=1; f12.shaderInt8=1; f12.storageBuffer8BitAccess=1; f12.vulkanMemoryModel=1;
    VkPhysicalDeviceVulkan11Features f11={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,.pNext=&f12}; f11.storageBuffer16BitAccess=1;
    VkPhysicalDeviceFeatures2 pf={VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&f11}; pf.features.shaderInt16=1; pf.features.shaderInt64=1;
    VkDeviceCreateInfo di={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&pf,.queueCreateInfoCount=1,.pQueueCreateInfos=&qi}; VkDevice d; C(vkCreateDevice(pd,&di,0,&d));
    VkShaderModuleCreateInfo si={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=n,.pCode=code}; VkShaderModule sm; C(vkCreateShaderModule(d,&si,0,&sm));
    VkDescriptorSetLayoutBinding b[8]; for(int i=0;i<nb;i++){b[i]=(VkDescriptorSetLayoutBinding){i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0};}
    VkDescriptorSetLayoutCreateInfo dl={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=nb,.pBindings=b}; VkDescriptorSetLayout dsl; C(vkCreateDescriptorSetLayout(d,&dl,0,&dsl));
    VkPushConstantRange pr={VK_SHADER_STAGE_COMPUTE_BIT,0,128}; VkPipelineLayoutCreateInfo pl={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&dsl,.pushConstantRangeCount=1,.pPushConstantRanges=&pr}; VkPipelineLayout lay; C(vkCreatePipelineLayout(d,&pl,0,&lay));
    VkSpecializationInfo spi={ns,me,4*ns,spec};
    VkComputePipelineCreateInfo cp={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sm,.pName="main",.pSpecializationInfo=&spi},.layout=lay}; VkPipeline pipe; C(vkCreateComputePipelines(d,0,1,&cp,0,&pipe));
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp); uint32_t mt=0; for(uint32_t i=0;i<mp.memoryTypeCount;i++) if(mp.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT){mt=i;break;}
    VkDescriptorBufferInfo dbi[8]; VkDeviceSize sz=1<<20; for(int i=0;i<nb;i++){ VkBufferCreateInfo bc={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=sz,.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}; VkBuffer bu; C(vkCreateBuffer(d,&bc,0,&bu)); VkMemoryAllocateInfo ma={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=sz,.memoryTypeIndex=mt}; VkDeviceMemory m; C(vkAllocateMemory(d,&ma,0,&m)); C(vkBindBufferMemory(d,bu,m,0)); void*pm; vkMapMemory(d,m,0,sz,0,&pm); memset(pm,0,sz); dbi[i]=(VkDescriptorBufferInfo){bu,0,sz}; }
    VkDescriptorPoolSize ps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,8}; VkDescriptorPoolCreateInfo dp={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&ps}; VkDescriptorPool pool; C(vkCreateDescriptorPool(d,&dp,0,&pool));
    VkDescriptorSetAllocateInfo da={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=pool,.descriptorSetCount=1,.pSetLayouts=&dsl}; VkDescriptorSet set; C(vkAllocateDescriptorSets(d,&da,&set));
    for(int i=0;i<nb;i++){VkWriteDescriptorSet w={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=set,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&dbi[i]}; vkUpdateDescriptorSets(d,1,&w,0,0);}
    VkCommandPoolCreateInfo cpi={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; VkCommandPool cpool; C(vkCreateCommandPool(d,&cpi,0,&cpool)); VkCommandBufferAllocateInfo cba={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=cpool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1}; VkCommandBuffer cb; C(vkAllocateCommandBuffers(d,&cba,&cb));
    VkCommandBufferBeginInfo bi={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(cb,&bi); vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe); vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,lay,0,1,&set,0,0); uint32_t pc[32]={0}; vkCmdPushConstants(cb,lay,VK_SHADER_STAGE_COMPUTE_BIT,0,128,pc); vkCmdDispatch(cb,1,1,1); vkEndCommandBuffer(cb);
    VkQueue q; vkGetDeviceQueue(d,0,0,&q); VkSubmitInfo sb={VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb}; C(vkQueueSubmit(q,1,&sb,0)); C(vkQueueWaitIdle(q)); printf("OK %s\n",av[1]); return 0;
}

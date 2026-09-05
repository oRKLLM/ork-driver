/* vk_bw_probe — Mali-G610 streaming memory bandwidth via Vulkan compute, with DDR-controller ground truth.
 *
 * The third-engine question: CPU + NPU together already reach ~29-30 GB/s and hold the DDR controller at
 * 88-90%, and pushing more demand at it (CPU3+NPU vs CPU2+NPU) stopped raising the total and only
 * redistributed it -- the signature of a saturated bus. If that ceiling is real, a Mali arm cannot add
 * throughput; it can only take a share. This probe measures what Mali alone can pull, so the prediction is
 * tested rather than assumed.
 *
 * CRITICAL: this board's Vulkan exposes TWO devices -- Mali-G610 and `llvmpipe`, a SOFTWARE rasteriser that
 * runs on the CPU. Selecting llvmpipe would measure CPU bandwidth and report it as GPU, which would be
 * worse than not measuring at all. The device is chosen by type + name and the choice is printed.
 *
 *   make vk_bw_probe && sudo ./vk_bw_probe [MiB] [seconds]
 */
#define _GNU_SOURCE
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e6+t.tv_nsec/1e3; }
#define VKC(x) do{ VkResult _r=(x); if(_r!=VK_SUCCESS){ printf("FAIL %s -> %d\n",#x,_r); return 1; } }while(0)

static volatile int g_dmc_stop=0; static volatile double g_dmc_avg=0,g_dmc_peak=0;
static void *dmc_sampler(void*a){
    cpu_set_t s; CPU_ZERO(&s); for(int i=0;i<4;i++) CPU_SET(i,&s); sched_setaffinity(0,sizeof s,&s);
    double sum=0; int n=0; g_dmc_peak=0;
    while(!g_dmc_stop){ FILE*f=fopen("/sys/class/devfreq/dmc/load","r");
        if(f){ int p=0; if(fscanf(f,"%d",&p)==1){ sum+=p; n++; if(p>g_dmc_peak) g_dmc_peak=p; } fclose(f); }
        struct timespec d={0,20*1000*1000}; nanosleep(&d,0); }
    g_dmc_avg=n?sum/n:0; (void)a; return NULL; }

int main(int argc,char**argv){
    size_t MIB = argc>1?(size_t)atoi(argv[1]):64;
    double SECS = argc>2?atof(argv[2]):3.0;
    int MODE = argc>3?atoi(argv[3]):0;          /* 0 = memory stream, 1 = compute-bound (no DRAM traffic) */
    uint32_t CITERS = argc>4?(uint32_t)atoi(argv[4]):20000;
    setvbuf(stdout,0,_IONBF,0);
    size_t BYTES = MIB<<20; uint32_t nvec = (uint32_t)(BYTES/16);   /* uvec4 = 16 B */

    VkInstance inst; VkApplicationInfo ai={VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion=VK_API_VERSION_1_1; ai.pApplicationName="vk_bw_probe";
    VkInstanceCreateInfo ici={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&ai;
    VKC(vkCreateInstance(&ici,0,&inst));

    uint32_t nd=0; vkEnumeratePhysicalDevices(inst,&nd,0);
    VkPhysicalDevice *pds=calloc(nd,sizeof *pds); vkEnumeratePhysicalDevices(inst,&nd,pds);
    VkPhysicalDevice pd=VK_NULL_HANDLE; VkPhysicalDeviceProperties props, chosen;
    for(uint32_t i=0;i<nd;i++){ vkGetPhysicalDeviceProperties(pds[i],&props);
        int is_sw = strstr(props.deviceName,"llvmpipe") || strstr(props.deviceName,"lavapipe")
                 || props.deviceType==VK_PHYSICAL_DEVICE_TYPE_CPU;
        printf("  vulkan device %u: %-28s type=%d %s\n",i,props.deviceName,props.deviceType,
               is_sw?"(SOFTWARE — rejected)":"");
        if(!is_sw && pd==VK_NULL_HANDLE){ pd=pds[i]; chosen=props; } }
    if(pd==VK_NULL_HANDLE){ printf("no hardware GPU found\n"); return 2; }
    printf("  -> using %s\n\n",chosen.deviceName);

    uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,0);
    VkQueueFamilyProperties*qf=calloc(nq,sizeof *qf); vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,qf);
    uint32_t qi=UINT32_MAX; for(uint32_t i=0;i<nq;i++) if(qf[i].queueFlags&VK_QUEUE_COMPUTE_BIT){ qi=i; break; }
    if(qi==UINT32_MAX){ printf("no compute queue\n"); return 2; }

    float prio=1.0f; VkDeviceQueueCreateInfo qci={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex=qi; qci.queueCount=1; qci.pQueuePriorities=&prio;
    VkDeviceCreateInfo dci={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
    VkDevice dev; VKC(vkCreateDevice(pd,&dci,0,&dev));
    VkQueue q; vkGetDeviceQueue(dev,qi,0,&q);

    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
    VkBuffer buf[2]; VkDeviceMemory mem[2]; size_t sz[2]={BYTES, 65536};
    for(int b=0;b<2;b++){
        VkBufferCreateInfo bci={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size=sz[b]; bci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        VKC(vkCreateBuffer(dev,&bci,0,&buf[b]));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,buf[b],&mr);
        uint32_t mt=UINT32_MAX; VkMemoryPropertyFlags want=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for(uint32_t i=0;i<mp.memoryTypeCount;i++)
            if((mr.memoryTypeBits&(1u<<i)) && (mp.memoryTypes[i].propertyFlags&want)==want){ mt=i; break; }
        if(mt==UINT32_MAX){ printf("no host-visible memory type\n"); return 2; }
        VkMemoryAllocateInfo mai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize=mr.size; mai.memoryTypeIndex=mt;
        VKC(vkAllocateMemory(dev,&mai,0,&mem[b]));
        VKC(vkBindBufferMemory(dev,buf[b],mem[b],0));
    }
    { void*p=0; VKC(vkMapMemory(dev,mem[0],0,BYTES,0,&p)); memset(p,1,BYTES); vkUnmapMemory(dev,mem[0]); }

    const char*spv = MODE ? "tools/vkcompute.spv" : "tools/vkbw.spv";
    const char*spv2= MODE ? "vkcompute.spv"       : "vkbw.spv";
    FILE*f=fopen(spv,"rb"); if(!f) f=fopen(spv2,"rb");
    if(!f){ printf("vkbw.spv not found (run: glslc -O tools/vkbw.comp -o tools/vkbw.spv)\n"); return 2; }
    fseek(f,0,SEEK_END); long ssz=ftell(f); fseek(f,0,SEEK_SET);
    uint32_t*code=malloc(ssz); if(fread(code,1,ssz,f)!=(size_t)ssz){ printf("spv read fail\n"); return 2; } fclose(f);
    VkShaderModuleCreateInfo smci={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize=ssz; smci.pCode=code;
    VkShaderModule sm; VKC(vkCreateShaderModule(dev,&smci,0,&sm));

    VkDescriptorSetLayoutBinding dsb[2]={{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
                                         {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0}};
    VkDescriptorSetLayoutCreateInfo dlci={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dlci.bindingCount=2; dlci.pBindings=dsb;
    VkDescriptorSetLayout dsl; VKC(vkCreateDescriptorSetLayout(dev,&dlci,0,&dsl));
    VkPushConstantRange pcr={VK_SHADER_STAGE_COMPUTE_BIT,0,4};
    VkPipelineLayoutCreateInfo plci={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount=1; plci.pSetLayouts=&dsl; plci.pushConstantRangeCount=1; plci.pPushConstantRanges=&pcr;
    VkPipelineLayout pl; VKC(vkCreatePipelineLayout(dev,&plci,0,&pl));
    VkComputePipelineCreateInfo cpci={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module=sm; cpci.stage.pName="main"; cpci.layout=pl;
    VkPipeline pipe; VKC(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpci,0,&pipe));

    VkDescriptorPoolSize dps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2};
    VkDescriptorPoolCreateInfo dpci={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpci.maxSets=1; dpci.poolSizeCount=1; dpci.pPoolSizes=&dps;
    VkDescriptorPool dp; VKC(vkCreateDescriptorPool(dev,&dpci,0,&dp));
    VkDescriptorSetAllocateInfo dsai={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsai.descriptorPool=dp; dsai.descriptorSetCount=1; dsai.pSetLayouts=&dsl;
    VkDescriptorSet ds; VKC(vkAllocateDescriptorSets(dev,&dsai,&ds));
    VkDescriptorBufferInfo dbi[2]={{buf[0],0,VK_WHOLE_SIZE},{buf[1],0,VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wds[2]={{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,ds,0,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,0,&dbi[0],0},
                                 {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,ds,1,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,0,&dbi[1],0}};
    vkUpdateDescriptorSets(dev,2,wds,0,0);

    VkCommandPoolCreateInfo cpi={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex=qi;
    VkCommandPool cp; VKC(vkCreateCommandPool(dev,&cpi,0,&cp));
    VkCommandBufferAllocateInfo cbai={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool=cp; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    VkCommandBuffer cb; VKC(vkAllocateCommandBuffers(dev,&cbai,&cb));

    uint32_t GROUPS = 256;
    VkCommandBufferBeginInfo cbbi={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKC(vkBeginCommandBuffer(cb,&cbbi));
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,0);
    uint32_t pcval = MODE ? CITERS : nvec;
    vkCmdPushConstants(cb,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&pcval);
    vkCmdDispatch(cb,GROUPS,1,1);
    VKC(vkEndCommandBuffer(cb));

    VkFenceCreateInfo fci={VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; VKC(vkCreateFence(dev,&fci,0,&fence));
    VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb;

    /* warm */
    for(int i=0;i<3;i++){ vkResetFences(dev,1,&fence); vkQueueSubmit(q,1,&si,fence);
        vkWaitForFences(dev,1,&fence,VK_TRUE,UINT64_MAX); }

    pthread_t dth; g_dmc_stop=0; pthread_create(&dth,0,dmc_sampler,0);
    double t0=now_us(); size_t bytes=0; int iters=0;
    while((now_us()-t0)/1e6 < SECS){
        vkResetFences(dev,1,&fence);
        if(vkQueueSubmit(q,1,&si,fence)!=VK_SUCCESS) break;
        vkWaitForFences(dev,1,&fence,VK_TRUE,UINT64_MAX);
        bytes+=BYTES; iters++;
    }
    double dt=(now_us()-t0)/1e6;
    g_dmc_stop=1; pthread_join(dth,0);
    if(MODE) printf("  Mali COMPUTE-bound: %.1f Mdispatch-iters/s  (%d dispatches x %u iters in %.2fs)\n",
                    (double)iters*CITERS*GROUPS*64/1e6/dt, iters, CITERS, dt);
    else     printf("  Mali streaming read: %.2f GB/s  (%d dispatches of %zu MiB in %.2fs)\n",bytes/1e9/dt,iters,MIB,dt);
    printf("  DMC during run: avg %.0f%%  peak %.0f%%   (theoretical peak 33.8 GB/s)\n",g_dmc_avg,g_dmc_peak);
    if(!MODE) printf("  implied bus share: %.0f%% of the ~30 GB/s practical ceiling\n", bytes/1e9/dt/30.0*100);
    return 0;
}

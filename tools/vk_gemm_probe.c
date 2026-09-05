/* vk_gemm_probe — Mali-G610 int8 GEMM throughput in GMAC/s, directly comparable to the NPU.
 *
 * The NPU measured 322 GMAC/s (645 GOP/s) on K=4096 N=1024 M=256 -- about 10.7% of its 6 TOPS nameplate,
 * because its matmul is weight-DMA-bound (arithmetic intensity is capped at the M-tile cap; HW WEIGHT_REUSE
 * is conv-only, Experiment Log #39). That leaves an open question for heterogeneous dispatch: is Mali a
 * genuine compute PEER, or a fallback? Answering it needs Mali on the SAME shape in the SAME units.
 *
 * This runs a real tiled int8 GEMM on Mali via the hardware 4x int8 dot product and VERIFIES the result
 * against a CPU reference before reporting any number -- a fast wrong GEMM is worthless, and an earlier
 * probe in this series did silently measure nothing (the compiler had deleted its workload).
 *
 *   make vk_gemm_probe && sudo ./vk_gemm_probe [M] [K] [N] [seconds]
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
    int M = argc>1?atoi(argv[1]):256, K = argc>2?atoi(argv[2]):4096, N = argc>3?atoi(argv[3]):1024;
    double SECS = argc>4?atof(argv[4]):3.0;
    setvbuf(stdout,0,_IONBF,0);
    if(M%64||N%64||K%16){ printf("M,N must be multiples of 64 and K of 16\n"); return 2; }
    size_t szA=(size_t)M*K, szB=(size_t)N*K, szC=(size_t)M*N*4;

    VkInstance inst; VkApplicationInfo ai={VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion=VK_API_VERSION_1_1; ai.pApplicationName="vk_gemm_probe";
    VkInstanceCreateInfo ici={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&ai;
    VKC(vkCreateInstance(&ici,0,&inst));
    uint32_t nd=0; vkEnumeratePhysicalDevices(inst,&nd,0);
    VkPhysicalDevice *pds=calloc(nd,sizeof *pds); vkEnumeratePhysicalDevices(inst,&nd,pds);
    VkPhysicalDevice pd=VK_NULL_HANDLE; VkPhysicalDeviceProperties props,chosen;
    for(uint32_t i=0;i<nd;i++){ vkGetPhysicalDeviceProperties(pds[i],&props);
        int sw = strstr(props.deviceName,"llvmpipe")||strstr(props.deviceName,"lavapipe")||props.deviceType==VK_PHYSICAL_DEVICE_TYPE_CPU;
        if(!sw && pd==VK_NULL_HANDLE){ pd=pds[i]; chosen=props; } }
    if(pd==VK_NULL_HANDLE){ printf("no hardware GPU\n"); return 2; }
    printf("  device: %s   shape M=%d K=%d N=%d\n",chosen.deviceName,M,K,N);

    uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,0);
    VkQueueFamilyProperties*qf=calloc(nq,sizeof *qf); vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,qf);
    uint32_t qi=UINT32_MAX; for(uint32_t i=0;i<nq;i++) if(qf[i].queueFlags&VK_QUEUE_COMPUTE_BIT){ qi=i; break; }
    float prio=1.f; VkDeviceQueueCreateInfo qci={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex=qi; qci.queueCount=1; qci.pQueuePriorities=&prio;
    const char*dext[]={"VK_KHR_shader_integer_dot_product","VK_KHR_8bit_storage"};
    VkDeviceCreateInfo dci={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci; dci.enabledExtensionCount=2; dci.ppEnabledExtensionNames=dext;
    VkDevice dev; VKC(vkCreateDevice(pd,&dci,0,&dev));
    VkQueue q; vkGetDeviceQueue(dev,qi,0,&q);

    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
    VkBuffer buf[3]; VkDeviceMemory mem[3]; void*map[3]; size_t sz[3]={szA,szB,szC};
    for(int b=0;b<3;b++){
        VkBufferCreateInfo bci={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size=sz[b]; bci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        VKC(vkCreateBuffer(dev,&bci,0,&buf[b]));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,buf[b],&mr);
        uint32_t mt=UINT32_MAX; VkMemoryPropertyFlags want=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for(uint32_t i=0;i<mp.memoryTypeCount;i++) if((mr.memoryTypeBits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&want)==want){ mt=i; break; }
        if(mt==UINT32_MAX){ printf("no host-visible memory\n"); return 2; }
        VkMemoryAllocateInfo mai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize=mr.size; mai.memoryTypeIndex=mt;
        VKC(vkAllocateMemory(dev,&mai,0,&mem[b]));
        VKC(vkBindBufferMemory(dev,buf[b],mem[b],0));
        VKC(vkMapMemory(dev,mem[b],0,sz[b],0,&map[b]));
    }
    int8_t *A=(int8_t*)map[0], *Bt=(int8_t*)map[1]; int32_t *C=(int32_t*)map[2];
    uint32_t s1=12345u; for(size_t i=0;i<szA;i++){ s1=s1*1664525u+1013904223u; A[i]=(int8_t)((s1>>16)&0x7f)-64; }
    uint32_t s2=6789u;  for(size_t i=0;i<szB;i++){ s2=s2*1664525u+1013904223u; Bt[i]=(int8_t)((s2>>16)&0x7f)-64; }
    memset(C,0,szC);

    FILE*f=fopen("tools/vkgemm.spv","rb"); if(!f) f=fopen("vkgemm.spv","rb");
    if(!f){ printf("vkgemm.spv missing (glslc -O tools/vkgemm.comp -o tools/vkgemm.spv)\n"); return 2; }
    fseek(f,0,SEEK_END); long ssz=ftell(f); fseek(f,0,SEEK_SET);
    uint32_t*code=malloc(ssz); if(fread(code,1,ssz,f)!=(size_t)ssz) return 2; fclose(f);
    VkShaderModuleCreateInfo smci={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize=ssz; smci.pCode=code;
    VkShaderModule sm; VKC(vkCreateShaderModule(dev,&smci,0,&sm));

    VkDescriptorSetLayoutBinding dsb[3];
    for(int i=0;i<3;i++){ dsb[i]=(VkDescriptorSetLayoutBinding){i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0}; }
    VkDescriptorSetLayoutCreateInfo dlci={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dlci.bindingCount=3; dlci.pBindings=dsb;
    VkDescriptorSetLayout dsl; VKC(vkCreateDescriptorSetLayout(dev,&dlci,0,&dsl));
    VkPushConstantRange pcr={VK_SHADER_STAGE_COMPUTE_BIT,0,12};
    VkPipelineLayoutCreateInfo plci={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount=1; plci.pSetLayouts=&dsl; plci.pushConstantRangeCount=1; plci.pPushConstantRanges=&pcr;
    VkPipelineLayout pl; VKC(vkCreatePipelineLayout(dev,&plci,0,&pl));
    VkComputePipelineCreateInfo cpci={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module=sm; cpci.stage.pName="main"; cpci.layout=pl;
    VkPipeline pipe; VKC(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpci,0,&pipe));

    VkDescriptorPoolSize dps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3};
    VkDescriptorPoolCreateInfo dpci={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpci.maxSets=1; dpci.poolSizeCount=1; dpci.pPoolSizes=&dps;
    VkDescriptorPool dp; VKC(vkCreateDescriptorPool(dev,&dpci,0,&dp));
    VkDescriptorSetAllocateInfo dsai={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsai.descriptorPool=dp; dsai.descriptorSetCount=1; dsai.pSetLayouts=&dsl;
    VkDescriptorSet ds; VKC(vkAllocateDescriptorSets(dev,&dsai,&ds));
    VkDescriptorBufferInfo dbi[3]; VkWriteDescriptorSet wds[3];
    for(int i=0;i<3;i++){ dbi[i]=(VkDescriptorBufferInfo){buf[i],0,VK_WHOLE_SIZE};
        wds[i]=(VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,ds,(uint32_t)i,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,0,&dbi[i],0}; }
    vkUpdateDescriptorSets(dev,3,wds,0,0);

    VkCommandPoolCreateInfo cpi={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex=qi;
    VkCommandPool cp; VKC(vkCreateCommandPool(dev,&cpi,0,&cp));
    VkCommandBufferAllocateInfo cbai={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool=cp; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    VkCommandBuffer cb; VKC(vkAllocateCommandBuffers(dev,&cbai,&cb));
    uint32_t pcv[3]={(uint32_t)M,(uint32_t)N,(uint32_t)K};
    VkCommandBufferBeginInfo cbbi={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKC(vkBeginCommandBuffer(cb,&cbbi));
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,0);
    vkCmdPushConstants(cb,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,12,pcv);
    vkCmdDispatch(cb,N/64,M/64,1);
    VKC(vkEndCommandBuffer(cb));
    VkFenceCreateInfo fci={VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; VKC(vkCreateFence(dev,&fci,0,&fence));
    VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb;

    /* ---- CORRECTNESS FIRST: strided sample across every output tile, exact int32 reference ---- */
    vkResetFences(dev,1,&fence); VKC(vkQueueSubmit(q,1,&si,fence));
    VKC(vkWaitForFences(dev,1,&fence,VK_TRUE,UINT64_MAX));
    long bad=0, checked=0; long stride=37;
    for(size_t idx=0; idx<(size_t)M*N; idx+=stride){
        int r=(int)(idx/N), cc=(int)(idx%N); int32_t ref=0;
        const int8_t *ar=A+(size_t)r*K, *br=Bt+(size_t)cc*K;
        for(int k=0;k<K;k++) ref += (int32_t)ar[k]*(int32_t)br[k];
        if(C[idx]!=ref){ if(bad<3) printf("  MISMATCH C[%d,%d] gpu=%d ref=%d\n",r,cc,C[idx],ref); bad++; }
        checked++;
    }
    printf("  verify: %ld/%ld sampled outputs exact%s\n", checked-bad, checked, bad?"  *** WRONG ***":"  (bit-exact)");
    if(bad){ printf("  refusing to report throughput for an incorrect GEMM.\n"); return 3; }

    for(int i=0;i<3;i++){ vkResetFences(dev,1,&fence); vkQueueSubmit(q,1,&si,fence); vkWaitForFences(dev,1,&fence,VK_TRUE,UINT64_MAX); }
    pthread_t dth; g_dmc_stop=0; pthread_create(&dth,0,dmc_sampler,0);
    double t0=now_us(); int it=0;
    while((now_us()-t0)/1e6 < SECS){ vkResetFences(dev,1,&fence);
        if(vkQueueSubmit(q,1,&si,fence)!=VK_SUCCESS) break;
        vkWaitForFences(dev,1,&fence,VK_TRUE,UINT64_MAX); it++; }
    double dt=(now_us()-t0)/1e6; g_dmc_stop=1; pthread_join(dth,0);
    double gmac = (double)it*M*N*K/1e9/dt;
    printf("  Mali int8 GEMM: %.1f GMAC/s (%.0f GOP/s)  %d iters in %.2fs   DMC avg %.0f%% peak %.0f%%\n",
           gmac, gmac*2, it, dt, g_dmc_avg, g_dmc_peak);
    printf("  compute density: %.0f MAC per DRAM byte\n", g_dmc_avg>0 ? gmac*1e9/(g_dmc_avg/100.0*33.8e9) : 0);
    return 0;
}

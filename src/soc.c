/* soc.c — runtime SoC detection from the device tree. */
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include "soc.h"

extern const struct ork_soc ork_soc_rk3588, ork_soc_rk3576;
/* registry — add new SoCs here (and a soc/<chip>.c) */
static const struct ork_soc *const TABLE[] = { &ork_soc_rk3588, &ork_soc_rk3576 };

const struct ork_soc *ork_soc_detect(void){
    /* /proc/device-tree/compatible is a list of NUL-separated "vendor,model" strings */
    char buf[512]; FILE *f=fopen("/proc/device-tree/compatible","r");
    if(!f) return NULL;
    size_t n=fread(buf,1,sizeof buf-1,f); fclose(f); buf[n]=0;
    for(size_t i=0;i<sizeof(TABLE)/sizeof(*TABLE);i++)
        for(size_t off=0; off<n; off+=strlen(buf+off)+1)
            if(strstr(buf+off, TABLE[i]->id)) return TABLE[i];
    return NULL;
}

/* soc.c — runtime SoC detection from the device tree. */
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include "soc.h"

extern const struct ork_soc ork_soc_rk3588, ork_soc_rk3576;
/* registry — add new SoCs here (and a soc/<chip>.c) */
static const struct ork_soc *const TABLE[] = { &ork_soc_rk3588, &ork_soc_rk3576 };

/* Look a SoC up by id instead of by device tree. This is what makes an OFFLINE context possible: a
 * machine that is not the board has no /proc/device-tree match, but the caps it needs (nmax, cbuf, core
 * count) are static data, not device state. Used by ork_npu_init_offline. */
const struct ork_soc *ork_soc_by_id(const char *id){
    if(!id) return NULL;
    for(size_t i=0;i<sizeof(TABLE)/sizeof(*TABLE);i++)
        if(!strcmp(id, TABLE[i]->id)) return TABLE[i];
    return NULL;
}

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

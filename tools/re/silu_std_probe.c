/* silu_std_probe.c — RE/calibration harness for the standalone on-NPU SiLU op (ork_i8_npu_probe_silu_std).
 * Phase 1 ("ramp"): load a ramp LUT (LUT[i]=i-512) and sweep the full int8 input range, printing the
 * transfer curve out(in) — this reveals the op's index math idx(in) so we can build a silu curve.
 * Phase 2 ("silu"): build a silu LUT for a chosen (in_scale,out_scale) under the measured index model and
 * validate the NPU output against a CPU silu reference.
 *
 *   cc -O2 -Iinclude -Isrc -pthread -o silu_std_probe tools/re/silu_std_probe.c src/npu.c src/soc.c \
 *       src/soc/rk3588.c src/soc/rk3576.c src/neon_activations.c -lm
 *   sudo ./silu_std_probe ramp <r_mult> <r_shift> <out_bias_hex> <idx_off_hex> <c4064_hex> <c4068_hex>
 *   sudo ./silu_std_probe silu <r_mult> <r_shift> <in_scale> <out_scale> <idx_off_hex> <c4064_hex> <c4068_hex>
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ork_npu.h"

static int clampi8(long v){ if(v>127)v=127; if(v<-128)v=-128; return (int)v; }
static int clampi16(long v){ if(v>32767)v=32767; if(v<-32768)v=-32768; return (int)v; }
static double siluf(double x){ return x/(1.0+exp(-x)); }
/* fp16 LUT stores fp16 bit patterns: encode a float as the raw 16-bit half it becomes */
static short f2hbits(double x){ union{ ork_f16 h; unsigned short u; } z; z.h=(ork_f16)x; return (short)z.u; }

int main(int argc, char**argv){
    if(argc<2){ printf("usage: %s ramp|silu ...\n",argv[0]); return 2; }
    ork_npu *c = ork_npu_init();
    if(!c){ printf("init failed (board only) — SKIP\n"); return 0; }
    if(!ork_ppu_fuse_enabled(c)){ printf("PPU not enabled — SKIP\n"); ork_npu_free(c); return 0; }

    const int M = getenv("PM")?atoi(getenv("PM")):8;
    const int N = getenv("PN")?atoi(getenv("PN")):64;
    static signed char in[8192]; static signed char out[8192];
    for(int i=0;i<M*N;i++) in[i]=(signed char)((i%256)-128);   /* sweep int8 range */
    static short lut[1030];

    if(!strcmp(argv[1],"ramp")){
        int r_mult = argc>2?atoi(argv[2]):0x4000, r_shift = argc>3?atoi(argv[3]):14;
        unsigned out_bias = argc>4?(unsigned)strtoul(argv[4],0,16):0;
        unsigned idx_off  = argc>5?(unsigned)strtoul(argv[5],0,16):0xffffc000;
        unsigned c4064    = argc>6?(unsigned)strtoul(argv[6],0,16):0;
        unsigned c4068    = argc>7?(unsigned)strtoul(argv[7],0,16):0;
        for(int i=0;i<1030;i++) lut[i]=(short)clampi16(i-512);   /* ramp: LUT[idx]=idx-512 */
        double us=0;
        int r = ork_i8_npu_probe_silu_std(c,in,M,N,r_mult,r_shift,out_bias,idx_off,c4064,c4068,lut,1030,out,&us);
        printf("ramp rc=%d (%.1f us)  r=0x%x/2^%d out_bias=0x%x idx_off=0x%x c4064=0x%x c4068=0x%x\n",
               r,us,r_mult,r_shift,out_bias,idx_off,c4064,c4068);
        if(r){ ork_npu_free(c); return 1; }
        for(int i=0;i<M*N;i++) if((i&7)==0||in[i]==0) printf("  in=%4d -> out=%4d\n", in[i], out[i]);
        ork_npu_free(c); return 0;
    }

    if(!strcmp(argv[1],"silu")){
        /* ork-native self-calibrating standalone SiLU: (1) measure idx(v) with a ramp LUT (R=1,bias=0), then
         * (2) build LUT[idx(v)] = silu(v*in_scale)/out_scale/R_run and run, validating vs CPU silu. */
        double in_scale = argc>2?atof(argv[2]):0.125, out_scale = argc>3?atof(argv[3]):0.063;
        int r_mult = argc>4?atoi(argv[4]):0x4000, r_shift = argc>5?atoi(argv[5]):14;
        unsigned idx_off = argc>6?(unsigned)strtoul(argv[6],0,16):0xffffc000;
        unsigned c4064   = argc>7?(unsigned)strtoul(argv[7],0,16):0xffff7dc8;
        unsigned c4068   = argc>8?(unsigned)strtoul(argv[8],0,16):0x411c0800;
        double Rrun=(double)r_mult/(double)(1u<<r_shift), us=0;
        /* pass 1: ramp LUT (LUT[i]=i-512), R=0.5, bias=0 -> out=clamp(0.5*(idx-512)) -> idx(v)=2*out+512.
         * R=0.5 keeps out unclamped across the full int8 input range (idx~[245,762]); mapping is R-independent. */
        for(int i=0;i<1030;i++) lut[i]=(short)clampi16(i-512);
        for(int i=0;i<M*N;i++) in[i]=(signed char)((i%256)-128);
        if(ork_i8_npu_probe_silu_std(c,in,M,N,0x2000,14,0,idx_off,c4064,c4068,lut,1030,out,&us)){ printf("calib wedged\n"); ork_npu_free(c); return 1; }
        int idxof[256]; for(int v=0;v<256;v++) idxof[v]=-1;
        for(int i=0;i<M*N;i++){ int v=(unsigned char)in[i]; int o=out[i];
            if(o>-127&&o<127) idxof[v]=2*o+512; }             /* skip saturated ends */
        /* pass 2: build silu curve at measured indices; interp gaps, hold ends */
        int set[1030]; for(int i=0;i<1030;i++){lut[i]=0;set[i]=0;}
        for(int vv=-128;vv<128;vv++){ int v=(unsigned char)vv; int idx=idxof[v]; if(idx<0||idx>1029)continue;
            double val=siluf(vv*in_scale)/out_scale/Rrun; lut[idx]=(short)clampi16(lround(val)); set[idx]=1; }
        int lo=-1,hi=-1; for(int i=0;i<1030;i++)if(set[i]){lo=i;break;} for(int i=1029;i>=0;i--)if(set[i]){hi=i;break;}
        for(int i=0;i<lo;i++)lut[i]=lut[lo]; for(int i=hi+1;i<1030;i++)lut[i]=lut[hi];
        for(int i=lo;i<=hi;i++){ if(set[i])continue; int a=i,b=i; while(a>lo&&!set[a])a--; while(b<hi&&!set[b])b++;
            lut[i]=(short)(lut[a]+(lut[b]-lut[a])*(i-a)/(b-a)); }
        /* pass 3: run with the built curve + chosen R, validate vs CPU silu */
        if(ork_i8_npu_probe_silu_std(c,in,M,N,r_mult,r_shift,0,idx_off,c4064,c4068,lut,1030,out,&us)){ printf("run wedged\n"); ork_npu_free(c); return 1; }
        int bad=0; double mx=0;
        for(int i=0;i<M*N;i++){ double ref=siluf(in[i]*in_scale)/out_scale; double got=out[i];
            double e=fabs(got-ref); if(e>mx)mx=e; if(e>2)bad++;
            if((i%37)==0) printf("  in=%4d out=%4d ref=%6.1f\n",in[i],out[i],ref); }
        printf("silu rc=0 (%.1f us) R=%.4f in_scale=%.3f out_scale=%.3f  bad(|e|>2)=%d/%d max|e|=%.2f\n",
               us,Rrun,in_scale,out_scale,bad,M*N,mx);
        ork_npu_free(c); return bad?1:0;
    }
    if(!strcmp(argv[1],"rampf16")){
        /* fp16 calibration: ramp LUT (int16 i-512), sweep fp16 inputs; reveals idx(in) + LUT->fp16 encoding */
        unsigned idx_off = argc>2?(unsigned)strtoul(argv[2],0,16):0xffffc000;
        unsigned c4064   = argc>3?(unsigned)strtoul(argv[3],0,16):0x80000000;
        unsigned c4068   = argc>4?(unsigned)strtoul(argv[4],0,16):0x69840000;
        double lo=argc>5?atof(argv[5]):-0.3, hi=argc>6?atof(argv[6]):0.3;
        int Mf=8,Nf=64; static ork_f16 inf[512],outf[512]; static short lut[1030];
        for(int i=0;i<1030;i++) lut[i]=f2hbits((double)(i-512));   /* LUT holds fp16 bits of (i-512) */
        for(int i=0;i<Mf*Nf;i++){ double x=lo+(hi-lo)*i/(Mf*Nf-1); inf[i]=(ork_f16)x; }
        double us=0;
        int r=ork_f16_npu_probe_silu_std(c,inf,Mf,Nf,idx_off,c4064,c4068,lut,1030,outf,&us);
        printf("rampf16 rc=%d (%.1f us) idx_off=0x%x c4064=0x%x c4068=0x%x\n",r,us,idx_off,c4064,c4068);
        if(r){ ork_npu_free(c); return 1; }
        for(int i=0;i<Mf*Nf;i+=16) printf("  in=%8.3f -> out=%8.3f\n",(double)(float)inf[i],(double)(float)outf[i]);
        ork_npu_free(c); return 0;
    }
    if(!strcmp(argv[1],"siluf16")){
        /* fp16 SiLU: measure idx(in) via ramp, build silu curve at those indices, run + validate vs CPU silu.
         * fp16 input IS the real value (no in_scale); out = silu(in). */
        unsigned idx_off=0xffffc000,c4064=0x80000000,c4068=0x69840000;
        double lo=argc>2?atof(argv[2]):-12.0, hi=argc>3?atof(argv[3]):12.0;
        int Mf=8,Nf=64; static ork_f16 inf[512],outf[512]; static short lut[1030];
        static double xv[512]; static int idxv[512];
        for(int i=0;i<1030;i++) lut[i]=(short)clampi16(i-512);
        for(int i=0;i<Mf*Nf;i++){ xv[i]=lo+(hi-lo)*i/(Mf*Nf-1); inf[i]=(ork_f16)xv[i]; }
        double us=0;
        if(ork_f16_npu_probe_silu_std(c,inf,Mf,Nf,idx_off,c4064,c4068,lut,1030,outf,&us)){ printf("calib wedged\n"); ork_npu_free(c); return 1; }
        int set[1030]; for(int i=0;i<1030;i++){lut[i]=0;set[i]=0;}
        for(int i=0;i<Mf*Nf;i++){ int idx=(int)lround((double)(float)outf[i])+512; idxv[i]=idx;
            if(idx>=0&&idx<1030){ double s=siluf(xv[i]); lut[idx]=(short)clampi16(lround(s*256.0)); set[idx]=1; } } /* scale silu by 256 in LUT */
        int flo=-1,fhi=-1; for(int i=0;i<1030;i++)if(set[i]){flo=i;break;} for(int i=1029;i>=0;i--)if(set[i]){fhi=i;break;}
        for(int i=0;i<flo;i++)lut[i]=lut[flo]; for(int i=fhi+1;i<1030;i++)lut[i]=lut[fhi];
        for(int i=flo;i<=fhi&&flo>=0;i++){ if(set[i])continue; int a=i,b=i; while(a>flo&&!set[a])a--; while(b<fhi&&!set[b])b++;
            lut[i]=(short)(lut[a]+(lut[b]-lut[a])*(i-a)/(b-a)); }
        if(ork_f16_npu_probe_silu_std(c,inf,Mf,Nf,idx_off,c4064,c4068,lut,1030,outf,&us)){ printf("run wedged\n"); ork_npu_free(c); return 1; }
        int bad=0; double mx=0;
        for(int i=0;i<Mf*Nf;i++){ double ref=siluf(xv[i]), got=(double)(float)outf[i];
            double e=fabs(got-ref); if(e>mx)mx=e; if(e>0.5)bad++;
            if((i%29)==0) printf("  in=%7.3f out=%8.4f ref=%8.4f\n",xv[i],got,ref); }
        printf("siluf16 rc=0 (%.1f us) bad(|e|>0.5)=%d/%d max|e|=%.3f\n",us,bad,Mf*Nf,mx);
        ork_npu_free(c); return bad?1:0;
    }
    if(!strcmp(argv[1],"sweep")){
        /* Fit idx = gain*in + offset for given c4064/c4068 on the int8 op (ramp LUT R=0.5 -> idx=2*out+512).
         * Non-verbose: prints one line so it can be called across a grid to decode the index-gain encoding. */
        unsigned c4064 = argc>2?(unsigned)strtoul(argv[2],0,16):0xffff7dc8;
        unsigned c4068 = argc>3?(unsigned)strtoul(argv[3],0,16):0x411c0800;
        unsigned idx_off = argc>4?(unsigned)strtoul(argv[4],0,16):0xffffc000;
        int idxof[256]; for(int v=0;v<256;v++)idxof[v]=-1;
        for(int i=0;i<M*N;i++) in[i]=(signed char)((i%256)-128);
        for(int i=0;i<1030;i++) lut[i]=(short)clampi16(i-512);
        double us=0;
        int r=ork_i8_npu_probe_silu_std(c,in,M,N,0x2000,14,0,idx_off,c4064,c4068,lut,1030,out,&us);
        if(r){ printf("c4064=%08x c4068=%08x WEDGED\n",c4064,c4068); ork_npu_free(c); return 1; }
        for(int i=0;i<M*N;i++){ int v=(unsigned char)in[i]; int o=out[i]; if(o>-127&&o<127) idxof[v]=2*o+512; }
        /* fit slope/intercept from unsaturated samples via least squares */
        double sx=0,sy=0,sxx=0,sxy=0; int n=0;
        for(int vv=-128;vv<128;vv++){ int idx=idxof[(unsigned char)vv]; if(idx<0)continue; sx+=vv;sy+=idx;sxx+=(double)vv*vv;sxy+=(double)vv*idx;n++; }
        if(n<3){ printf("c4064=%08x c4068=%08x  (n=%d too few)\n",c4064,c4068,n); ork_npu_free(c); return 0; }
        double gain=(n*sxy-sx*sy)/(n*sxx-sx*sx), off=(sy-gain*sx)/n;
        int vlo=999,vhi=-999; for(int vv=-128;vv<128;vv++)if(idxof[(unsigned char)vv]>=0){ if(vv<vlo)vlo=vv; if(vv>vhi)vhi=vv; }
        printf("c4064=%08x c4068=%08x  gain=%.4f offset=%.1f  (n=%d, in[%d..%d])\n",c4064,c4068,gain,off,n,vlo,vhi);
        ork_npu_free(c); return 0;
    }
    if(!strcmp(argv[1],"sweepi16")){
        /* Fit idx = gain*in + offset on the INT16 op (ramp LUT R=1 -> out=idx-512, int16 unclamped -> idx=out+512).
         * Sweep c4068 to find the gain-1 value (integer idx => bit-exact). Inputs [-A,A]. */
        unsigned c4064 = argc>2?(unsigned)strtoul(argv[2],0,16):0xffff7dc8;
        unsigned c4068 = argc>3?(unsigned)strtoul(argv[3],0,16):0x411c1000;
        int A = argc>4?atoi(argv[4]):500;
        unsigned idx_off = argc>5?(unsigned)strtoul(argv[5],0,16):0xffffc000;
        int Mf=16,Nf=64; static short in16[1024],out16[1024]; static short lut[1030];
        for(int i=0;i<Mf*Nf;i++) in16[i]=(short)(-A + (2*A)*i/(Mf*Nf-1));
        for(int i=0;i<1030;i++) lut[i]=(short)clampi16(i-512);
        double us=0;
        int r=ork_i16_npu_probe_silu_std(c,in16,Mf,Nf,0x4000,14,0,idx_off,c4064,c4068,lut,1030,out16,&us);
        if(r){ printf("c4064=%08x c4068=%08x WEDGED\n",c4064,c4068); ork_npu_free(c); return 1; }
        double sx=0,sy=0,sxx=0,sxy=0; int n=0;
        for(int i=0;i<Mf*Nf;i++){ int o=out16[i]; if(o<=-32000||o>=32000)continue; double x=in16[i],y=o+512; sx+=x;sy+=y;sxx+=x*x;sxy+=x*y;n++; }
        if(n<3){ printf("c4064=%08x c4068=%08x (n=%d)\n",c4064,c4068,n); ork_npu_free(c); return 0; }
        double gain=(n*sxy-sx*sy)/(n*sxx-sx*sx), off=(sy-gain*sx)/n;
        printf("idx_off=%08x c4064=%08x c4068=%08x  gain=%.5f offset=%.1f  (n=%d)\n",idx_off,c4064,c4068,gain,off,n);
        ork_npu_free(c); return 0;
    }
    if(!strcmp(argv[1],"silui16x")){
        /* BIT-EXACT int16 attempt: (1) measure FRACTIONAL idx(v) via a linear ramp at high R (linear ramp is
         * interpolated exactly, so out=R*(idx-512) reveals idx to 1/R). (2) Solve the LUT by Gauss-Seidel so the
         * op's 6-bit linear interp lands on silu(v) at each input's exact fractional idx. (3) run at R=1, validate.
         * Uses gain-1.24 params; idx assumed R-independent (verified for int8). */
        double in_scale=argc>2?atof(argv[2]):0.03, out_scale=argc>3?atof(argv[3]):0.03;
        unsigned idx_off=argc>4?(unsigned)strtoul(argv[4],0,16):0xffffe000;
        unsigned c4068=argc>5?(unsigned)strtoul(argv[5],0,16):0x411c0200;
        unsigned c4064=0xffff7dc8;
        int Mf=16,Nf=64; static short in16[1024],out16[1024]; static short lut[1030];
        static double Iv[1024], Tv[1024]; int nv=0;
        for(int i=0;i<Mf*Nf;i++) in16[i]=(short)(i-512);         /* inputs -512..511, one each */
        /* (1) fractional-idx calibration: linear ramp LUT (interpolated EXACTLY), R=64 -> out=64*(idx-512) */
        for(int i=0;i<1030;i++) lut[i]=(short)clampi16(i-512);
        double us=0;
        if(ork_i16_npu_probe_silu_std(c,in16,Mf,Nf,0x4000,8,0,idx_off,c4064,c4068,lut,1030,out16,&us)){ printf("calib wedged\n"); ork_npu_free(c); return 1; }
        double gsum=0; int gc=0; double prevI=-1;
        for(int i=0;i<Mf*Nf;i++){ int v=in16[i]; double I=out16[i]/64.0+512.0; if(I<2||I>1027)continue;
            Iv[nv]=I; Tv[nv]=siluf(v*in_scale)/out_scale; if(prevI>0){gsum+=fabs(I-prevI);gc++;} prevI=I; nv++; }
        double gain=gc?gsum/gc:0;
        /* (2) solve LUT via Gauss-Seidel so op interp at each measured fractional idx hits silu(v) */
        double flut[1030]; for(int k=0;k<1030;k++) flut[k]=0;
        for(int j=0;j<nv;j++){ int k=(int)lround(Iv[j]); if(k>=0&&k<1030) flut[k]=Tv[j]; }  /* seed */
        for(int k=1;k<1030;k++) if(flut[k]==0&&flut[k-1]!=0) flut[k]=flut[k-1];
        for(int it=0;it<500;it++){ double maxr=0;
            for(int j=0;j<nv;j++){ int f=(int)floor(Iv[j]); double fr=Iv[j]-f; if(f<0||f+1>=1030)continue;
                double pred=flut[f]*(1-fr)+flut[f+1]*fr, r=Tv[j]-pred; if(fabs(r)>maxr)maxr=fabs(r);
                double w=(1-fr)*(1-fr)+fr*fr; if(w<1e-9)continue;
                flut[f]+=r*(1-fr)/w; flut[f+1]+=r*fr/w; }
            if(maxr<0.005)break; }
        for(int k=0;k<1030;k++){ long q=lround(flut[k]); if(q>32767)q=32767; if(q<-32768)q=-32768; lut[k]=(short)q; }
        /* (3) run at R=1, validate over the in-range inputs */
        if(ork_i16_npu_probe_silu_std(c,in16,Mf,Nf,0x4000,14,0,idx_off,c4064,c4068,lut,1030,out16,&us)){ printf("run wedged\n"); ork_npu_free(c); return 1; }
        int bad=0; long mx=0; int checked=0;
        for(int i=0;i<Mf*Nf;i++){ int v=in16[i]; double I=64.0; /* recompute in-range via idx from calib not avail; use target range */
            double ref_d=siluf(v*in_scale)/out_scale; long ref=lround(ref_d); if(ref>32767)ref=32767; if(ref<-32768)ref=-32768;
            /* only score inputs whose idx was in range (approx: |out| not saturated) */
            if(out16[i]<=-32760||out16[i]>=32760) continue; checked++;
            long e=labs((long)out16[i]-ref); if(e>mx)mx=e; if(e>1)bad++;
            if((i%61)==0) printf("  in=%5d out=%6d ref=%6ld\n",v,out16[i],ref); (void)I; }
        printf("silui16x rc=0 (%.1f us) idx_off=%08x c4068=%08x gain=%.3f in_scale=%.3f  nv=%d checked=%d bad(|e|>1)=%d max|e|=%ld\n",
               us,idx_off,c4068,gain,in_scale,nv,checked,bad,mx);
        ork_npu_free(c); return bad?1:0;
    }
    if(!strcmp(argv[1],"silui16")){
        /* int16 SiLU: measure idx(in) via ramp (R=0.5), build silu curve, run + validate vs CPU. Inputs span
         * [-A,A] int16; in_scale maps int16->real, out_scale maps real->int16 output. */
        double in_scale=argc>2?atof(argv[2]):0.0625, out_scale=argc>3?atof(argv[3]):0.0625;
        int A=argc>4?atoi(argv[4]):112;
        unsigned idx_off=argc>5?(unsigned)strtoul(argv[5],0,16):0xffffc000;
        unsigned c4064=argc>6?(unsigned)strtoul(argv[6],0,16):0xffff7dc8;  /* default: int8's proven index gain */
        unsigned c4068=argc>7?(unsigned)strtoul(argv[7],0,16):0x411c0800;
        int Mf=8,Nf=64; static short in16[512],out16[512]; static short lut[1030]; static int idxof[512];
        for(int i=0;i<Mf*Nf;i++) in16[i]=(short)(-A + (2*A)*i/(Mf*Nf-1));
        for(int i=0;i<1030;i++) lut[i]=(short)clampi16(i-512);
        double us=0;
        /* int16 output isn't clamped at 127, so calibrate AND run at R=1 (out=idx-512 -> idx=out+512) */
        if(ork_i16_npu_probe_silu_std(c,in16,Mf,Nf,0x4000,14,0,idx_off,c4064,c4068,lut,1030,out16,&us)){ printf("calib wedged\n"); ork_npu_free(c); return 1; }
        int set[1030]; for(int i=0;i<1030;i++){lut[i]=0;set[i]=0;}
        for(int i=0;i<Mf*Nf;i++){ int o=out16[i]; idxof[i]=(o>-32000&&o<32000)?o+512:-1; int idx=idxof[i];
            if(idx>=0&&idx<1030){ double s=siluf(in16[i]*in_scale)/out_scale; lut[idx]=(short)clampi16(lround(s)); set[idx]=1; } }
        int flo=-1,fhi=-1; for(int i=0;i<1030;i++)if(set[i]){flo=i;break;} for(int i=1029;i>=0;i--)if(set[i]){fhi=i;break;}
        if(flo<0){ printf("no idx measured\n"); ork_npu_free(c); return 1; }
        for(int i=0;i<flo;i++)lut[i]=lut[flo]; for(int i=fhi+1;i<1030;i++)lut[i]=lut[fhi];
        for(int i=flo;i<=fhi;i++){ if(set[i])continue; int a=i,b=i; while(a>flo&&!set[a])a--; while(b<fhi&&!set[b])b++;
            lut[i]=(short)(lut[a]+(lut[b]-lut[a])*(i-a)/(b-a)); }
        if(ork_i16_npu_probe_silu_std(c,in16,Mf,Nf,0x4000,14,0,idx_off,c4064,c4068,lut,1030,out16,&us)){ printf("run wedged\n"); ork_npu_free(c); return 1; }
        int bad=0; double mx=0;
        for(int i=0;i<Mf*Nf;i++){ double ref=siluf(in16[i]*in_scale)/out_scale, got=out16[i];
            double e=fabs(got-ref); if(e>mx)mx=e; if(e>2)bad++;
            if((i%37)==0) printf("  in=%6d out=%6d ref=%8.1f\n",in16[i],out16[i],ref); }
        printf("silui16 rc=0 (%.1f us) in_scale=%.4f out_scale=%.4f A=%d bad(|e|>2)=%d/%d max|e|=%.2f\n",
               us,in_scale,out_scale,A,bad,Mf*Nf,mx);
        ork_npu_free(c); return bad?1:0;
    }
    printf("unknown mode\n"); ork_npu_free(c); return 2;
}

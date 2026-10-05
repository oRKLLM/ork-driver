/* gdn_fixture_check — check the C chunkwise GDN backward against MLX fp32 fixtures, at real width.
 *
 * gdn_chunk_bwd.c settles the ALGEBRA: it agrees with the sequential recurrence to ~1e-14 in fp64,
 * at every chunk length. That says nothing about NUMERICS at the shape the model actually runs
 * (d=128, T=1024) in the precision it actually runs (fp32 accumulators, fp16 operands), where the
 * live quantities are gamma_t/gamma_s ratios spanning orders of magnitude and a triangular solve.
 * This checks the same implementation against a reference that is already training models.
 *
 * It INCLUDES gdn_chunk_bwd.c rather than reimplementing it — one implementation, compiled at two
 * precisions (-DGDN_REAL=float / double). A second copy would drift from the first, which is the
 * failure mode the whole GDN exercise has been built to avoid.
 *
 * The fixture comes from the MLX side, which is where MLX is installed — ork-driver is C11 with no
 * Python and no third-party libraries, so the interchange is a flat file this reads with fgets and
 * fread:
 *     # in qwen35-lora-moe
 *     .venv/bin/python tools/export_gdn_fixtures.py --T 1024 --dk 128 --dv 128 --out /tmp/gdn.fix
 *     # here
 *     cc -O2 -o /tmp/gfc tools/gdn_fixture_check.c -lm && /tmp/gfc /tmp/gdn.fix
 *
 * The fixture carries TWO sets of gradients — from MLX autograd through the sequential reference,
 * and from the production chunkwise custom VJP — so a disagreement with both is a porting error
 * while a disagreement with one is a numerical one. Its header also carries SELFCONSIST, how well
 * those two agree on the exporting host, which is the floor on anything this can claim: on a host
 * whose fp32 matmul is really bf16 (measured 8.3e-04 on an M5 Max against 2.7e-07 on an M3 Max),
 * no implementation can agree with the fixture better than the fixture agrees with itself.
 */
#define GDN_CHUNK_NO_MAIN 1
/* The included file also carries its own self-test helpers, which this consumer does not call. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "gdn_chunk_bwd.c"
#pragma GCC diagnostic pop

#define MAXARR 32
typedef struct { char name[32]; int ndim, dim[4]; size_t n, off; } arrinfo;

static arrinfo AR[MAXARR];
static int NAR;
static float *BLOB;

static const float *find(const char*nm, size_t want){
    for(int i=0;i<NAR;i++) if(!strcmp(AR[i].name,nm)){
        if(want && AR[i].n!=want){
            fprintf(stderr,"fixture array %s has %zu floats, expected %zu\n",nm,AR[i].n,want);
            exit(1);
        }
        return BLOB+AR[i].off;
    }
    fprintf(stderr,"fixture is missing array %s\n",nm); exit(1);
}

/* max|a-b| / max|b|, the same measure gdn_chunk_bwd.c uses. */
static double relmax(const real*a,const float*b,size_t n){
    double w=0,m=0;
    for(size_t i=0;i<n;i++){ double d=fabs((double)a[i]-(double)b[i]); if(d>w) w=d;
                             if(fabs((double)b[i])>m) m=fabs((double)b[i]); }
    return m>0? w/m : w;
}
int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s <fixture> [chunk-override]\n",argv[0]); return 2; }
    FILE*f=fopen(argv[1],"rb");
    if(!f){ perror(argv[1]); return 2; }

    char line[512]; int CH=0, degraded=0; double selfc=-1; char host[64]="?", mlxv[32]="?";
    size_t total=0;
    if(!fgets(line,sizeof line,f) || strncmp(line,"ORKGDNFIX 1",11)){
        fprintf(stderr,"not an ORKGDNFIX v1 file\n"); return 2; }
    while(fgets(line,sizeof line,f)){
        if(!strncmp(line,"DATA",4)) break;
        char key[32];
        if(sscanf(line,"%31s",key)!=1) continue;
        if     (!strcmp(key,"T"))      sscanf(line,"T %d",&T);
        else if(!strcmp(key,"DK"))     sscanf(line,"DK %d",&DK);
        else if(!strcmp(key,"DV"))     sscanf(line,"DV %d",&DV);
        else if(!strcmp(key,"CHUNK"))  sscanf(line,"CHUNK %d",&CH);
        else if(!strcmp(key,"MATMUL_DEGRADED")) sscanf(line,"MATMUL_DEGRADED %d",&degraded);
        else if(!strcmp(key,"SELFCONSIST"))     sscanf(line,"SELFCONSIST %lf",&selfc);
        else if(!strcmp(key,"HOST"))   sscanf(line,"HOST %63s",host);
        else if(!strcmp(key,"MLX"))    sscanf(line,"MLX %31s",mlxv);
        else if(!strcmp(key,"ARRAY")){
            arrinfo*a=&AR[NAR];
            char rest[256]="";
            if(sscanf(line,"ARRAY %31s %255[^\n]",a->name,rest)<1) continue;
            a->n=1; a->ndim=0;
            for(char*t=strtok(rest," ");t && a->ndim<4;t=strtok(NULL," ")){
                a->dim[a->ndim]=atoi(t); a->n*=(size_t)a->dim[a->ndim]; a->ndim++;
            }
            a->off=total; total+=a->n;
            if(++NAR>=MAXARR){ fprintf(stderr,"too many arrays\n"); return 2; }
        }
    }
    if(argc>2) CH=atoi(argv[2]);
    if(T<=0||DK<=0||DV<=0||CH<=0){ fprintf(stderr,"bad header\n"); return 2; }

    BLOB=malloc(total*sizeof(float));
    if(!BLOB || fread(BLOB,sizeof(float),total,f)!=total){
        fprintf(stderr,"short read: wanted %zu floats\n",total); return 2; }
    fclose(f);

    printf("fixture %s\n  T=%d dk=%d dv=%d chunk=%d   host %s, mlx %s%s\n",
           argv[1],T,DK,DV,CH,host,mlxv, degraded?"   [fp32 matmul DEGRADED on that host]":"");
    printf("  the fixture's own two paths agree to %.3e -- nothing below can beat that\n\n", selfc);

    size_t tk=(size_t)T*DK, tv=(size_t)T*DV, sz=(size_t)DV*DK;

    /* Load the fixture's inputs into the implementation's globals, at its compiled precision. */
    #define LOAD(dst,nm,n) do{ const float*p_=find(nm,n); dst=malloc((n)*RSZ); \
        for(size_t i_=0;i_<(size_t)(n);i_++) dst[i_]=(real)p_[i_]; }while(0)
    LOAD(q,"q",tk); LOAD(k,"k",tk); LOAD(v,"v",tv);
    LOAD(g,"g",(size_t)T); LOAD(bet,"beta",(size_t)T); LOAD(S0,"state",sz);
    LOAD(dY,"dy",tv); LOAD(dSo,"dstate",sz);
    Y=malloc(tv*RSZ); Sf=malloc(sz*RSZ);
    dq=malloc(tk*RSZ); dk=malloc(tk*RSZ); dv=malloc(tv*RSZ);
    dlg=malloc((size_t)T*RSZ); dbe=malloc((size_t)T*RSZ); dSi=malloc(sz*RSZ);

    /* The fixture's dg is d/dg; this implementation produces d/dlog g. Convert, don't compare
     * the wrong quantity -- they differ by a factor g, which is 1e-3 at the small decays the real
     * parameterisation produces and would look like a catastrophic failure. */
    real *dgc=malloc((size_t)T*RSZ);

    fwd_chunked(CH);
    bwd_chunked(CH);
    for(int t=0;t<T;t++) dgc[t]=dlg[t]/g[t];

    struct { const char*nm; const real*got; const char*seq; const char*chk; size_t n; } rows[] = {
        {"y",         Y,   "y_seq",         "y_chunk",         tv},
        {"state_out", Sf,  "state_out_seq", "state_out_chunk", sz},
        {"dq",        dq,  "dq_seq",        "dq_chunk",        tk},
        {"dk",        dk,  "dk_seq",        "dk_chunk",        tk},
        {"dv",        dv,  "dv_seq",        "dv_chunk",        tv},
        {"dg",        dgc, "dg_seq",        "dg_chunk",        (size_t)T},
        {"dbeta",     dbe, "dbeta_seq",     "dbeta_chunk",     (size_t)T},
        {"dstate_in", dSi, "dstate_in_seq", "dstate_in_chunk", sz},
    };

    /* The bar: the fixture's own self-consistency, with room for this side's precision. An fp32
     * build cannot do better than ~1e-6 relative on accumulations this long, so take the larger. */
    double bar = selfc>0? selfc*8 : 1e-5;
    double fp  = (RSZ==sizeof(float))? 2e-5 : 0.0;
    if(fp>bar) bar=fp;

    printf("%-11s  %-12s  %-12s  %s\n","", "vs sequential","vs chunkwise","");
    int bad=0;
    for(unsigned i=0;i<sizeof rows/sizeof*rows;i++){
        double rs=relmax(rows[i].got, find(rows[i].seq,rows[i].n), rows[i].n);
        double rc=relmax(rows[i].got, find(rows[i].chk,rows[i].n), rows[i].n);
        int f1=rs>bar, f2=rc>bar;
        const char*verdict = (!f1&&!f2) ? "PASS"
                           : ( f1&& f2) ? "FAIL -- disagrees with BOTH: a porting error"
                           :              "FAIL -- agrees with one path only: numerical";
        printf("  %-9s  %.3e     %.3e     %s\n", rows[i].nm, rs, rc, verdict);
        bad += (f1||f2);
    }
    printf("\nbar %.1e  (%s, %zu-byte reals)\n", bar,
           RSZ==sizeof(float)?"fp32 build":"fp64 build", RSZ);
    printf("%s\n", bad==0
      ? "ALL PASS -- the C chunkwise backward matches MLX at this width and precision."
      : "SOME FAILED -- see the verdict column for which kind.");
    return bad!=0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <quda_internal.h>

#include <quda.h>
#include <gauge_quda.h>
#include <dslash_quda.h>
#include <llfat_quda.h>

#include <test_util.h>
#include <llfat_reference.h>
#include "misc.h"
#include <cuda.h>

#include "face_quda.h"
#include "mpicomm.h"
#include <mpi.h>

#define MAX(a,b) ((a)>(b)? (a):(b))
#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))


static FullGauge cudaSiteLink, cudaSiteLink_ex, cudaSiteLink_nl;
static FullGauge cudaFatLink;
static FullStaple cudaStaple, cudaStaple_ex, cudaStaple_nl;
static FullStaple cudaStaple1, cudaStaple1_ex, cudaStaple1_nl;
QudaGaugeParam gaugeParam;
QudaGaugeParam gaugeParam_ex;
QudaGaugeParam gaugeParam_nl;
void *fatlink, *sitelink[4], *reflink[4];

void* ghost_sitelink[4];
void* ghost_sitelink_diag[16];
void* sitelink_ex[4];
void* sitelink_nl[4];


int verify_results = 0;

extern void initDslashCuda(FullGauge gauge);

#define DIM 24

int device = 0;
int ODD_BIT = 1;
int Z[4];
int V;
int Vh;
int Vs[4];
int Vsh[4];
int Vs_x, Vs_y, Vs_z, Vs_t;
int Vsh_x, Vsh_y, Vsh_z, Vsh_t;

int Z_ex[4];
int V_ex;
int Vh_ex;
int Vs_ex[4];
int Vsh_ex[4];
int Vs_ex_x, Vs_ex_y, Vs_ex_z, Vs_ex_t;
int Vsh_ex_x, Vsh_ex_y, Vsh_ex_z, Vsh_ex_t;

int X1, X1h, X2, X3, X4;
int E1, E1h, E2, E3, E4;
int L1, L1h, L2, L3, L4;
int Vh_nl;

extern int xdim, ydim, zdim, tdim;
extern int gridsize_from_cmdline[];

extern QudaReconstructType link_recon;
extern QudaPrecision  prec;
QudaPrecision  cpu_prec = QUDA_DOUBLE_PRECISION;
size_t gSize;

typedef struct {
  double real;
  double imag;
} dcomplex;

typedef struct { dcomplex e[3][3]; } dsu3_matrix;



void
setDims(int *X) {
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    Z[d] = X[d];
  }
  Vh = V/2;

  Vs[0] = Vs_x = X[1]*X[2]*X[3];
  Vs[1] = Vs_y = X[0]*X[2]*X[3];
  Vs[2] = Vs_z = X[0]*X[1]*X[3];
  Vs[3] = Vs_t = X[0]*X[1]*X[2];

  Vsh[0] = Vsh_x = Vs_x/2;
  Vsh[1] = Vsh_y = Vs_y/2;
  Vsh[2] = Vsh_z = Vs_z/2;
  Vsh[3] = Vsh_t = Vs_t/2;

  V_ex = 1;
  for (int d=0; d< 4; d++) {
    V_ex *= X[d]+4;
    Z_ex[d] = X[d]+4;
  }
  Vh_ex = V_ex/2;
  
  Vs_ex[0] = Vs_ex_x = Z_ex[1]*Z_ex[2]*Z_ex[3];
  Vs_ex[1] = Vs_ex_y = Z_ex[0]*Z_ex[2]*Z_ex[3];
  Vs_ex[2] = Vs_ex_z = Z_ex[0]*Z_ex[1]*Z_ex[3];
  Vs_ex[3] = Vs_ex_t = Z_ex[0]*Z_ex[1]*Z_ex[2];
  
  Vsh_ex[0] = Vsh_ex_x = Vs_ex_x/2;
  Vsh_ex[1] = Vsh_ex_y = Vs_ex_y/2;
  Vsh_ex[2] = Vsh_ex_z = Vs_ex_z/2;
  Vsh_ex[3] = Vsh_ex_t = Vs_ex_t/2;


  X1=X[0]; X2 = X[1]; X3=X[2]; X4=X[3];
  X1h=X1/2;
  E1=X1+4; E2=X2+4; E3=X3+4; E4=X4+4;
  E1h=E1/2;
  
  L1 = X[0] + 2;
  L2 = X[0] + 2;
  L3 = X[0] + 2;
  L4 = X[0] + 2;
  L1h = L1/2;
  
  Vh_nl = L1*L2*L3*L4/2;
}

static void
llfat_init(int test)
{ 
  initQuda(device);
  //cudaSetDevice(dev); CUERR;
    
  gaugeParam.X[0] = xdim;
  gaugeParam.X[1] = ydim;
  gaugeParam.X[2] = zdim;
  gaugeParam.X[3] = tdim;

  setDims(gaugeParam.X);
    
  gaugeParam.cpu_prec = cpu_prec;
  gaugeParam.cuda_prec = prec;
        
  gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
    
  int i;
#if (CUDA_VERSION >=4000)
  cudaMallocHost((void**)&fatlink,  4*V*gaugeSiteSize*gSize);
#else
  fatlink = malloc(4*V*gaugeSiteSize*gSize);
#endif
  if (fatlink == NULL){
    fprintf(stderr, "ERROR: malloc failed for fatlink\n");
    exit(1);
  }
  
  for(i=0;i < 4;i++){
#if (CUDA_VERSION >=4000)
    cudaMallocHost((void**)&sitelink[i], V*gaugeSiteSize* gSize);
#else
    sitelink[i] = malloc(V*gaugeSiteSize* gSize);
#endif
    if (sitelink[i] == NULL){
      fprintf(stderr, "ERROR: malloc failed for sitelink[%d]\n", i);
      exit(1);
    }
  }

  for(i=0;i < 4;i++){
    reflink[i] = malloc(V*gaugeSiteSize* gSize);
    if (reflink[i] == NULL){
      fprintf(stderr, "ERROR: malloc failed for reflink[%d]\n", i);
      exit(1);
    }
  }

  //we need x,y,z site links in the back and forward T slice
  // so it is 3*2*Vs_t
  for(i=0;i < 4; i++){
    ghost_sitelink[i] = malloc(8*Vs[i]*gaugeSiteSize*gSize);
    if (ghost_sitelink[i] == NULL){
	printf("ERROR: malloc failed for ghost_sitelink[%d] \n",i);
	exit(1);
    }
  }
  
  /*
   * nu |     |
   *    |_____|
   *      mu     
   */
  
  for(int nu=0;nu < 4;nu++){
    for(int mu=0; mu < 4;mu++){
	if(nu == mu){
	  ghost_sitelink_diag[nu*4+mu] = NULL;
	}else{
	  //the other directions
	  int dir1, dir2;
	  for(dir1= 0; dir1 < 4; dir1++){
	    if(dir1 !=nu && dir1 != mu){
	      break;
	    }
	  }
	  for(dir2=0; dir2 < 4; dir2++){
	    if(dir2 != nu && dir2 != mu && dir2 != dir1){
	      break;
	    }
	  }
	  ghost_sitelink_diag[nu*4+mu] = malloc(Z[dir1]*Z[dir2]*gaugeSiteSize*gSize);
	  if(ghost_sitelink_diag[nu*4+mu] == NULL){
	  errorQuda("malloc failed for ghost_sitelink_diag\n");
	  }
	  
	  memset(ghost_sitelink_diag[nu*4+mu], 0, Z[dir1]*Z[dir2]*gaugeSiteSize*gSize);
	}	
    }
  }

  createSiteLinkCPU(sitelink, gaugeParam.cpu_prec, 1);
  
  /*
  {
    double* data = (double*)sitelink[1];
    data += (Vh) * gaugeSiteSize;
    printf("cpu sitelink=\n");
    printf("(%f %f) (%f %f) (%f %f)\n"
	   "(%f %f) (%f %f) (%f %f)\n"
	   "(%f %f) (%f %f) (%f %f)\n",
	   data[0], data[1], data[2], data[3], data[4], data[5], 
	   data[6], data[7], data[8], data[9], data[10], data[11], 
	   data[12], data[13], data[14], data[15], data[16], data[17]); 
  }
*/


  for(i=0;i < 4;i++){
#if (CUDA_VERSION >=4000)
    cudaMallocHost((void**)&sitelink_ex[i], V_ex*gaugeSiteSize* gSize);
    cudaMemset(sitelink_ex[i], 0, V_ex*gaugeSiteSize* gSize);
#else
    sitelink_ex[i] = malloc(V_ex*gaugeSiteSize* gSize);
#endif
    if (sitelink_ex[i] == NULL){
      fprintf(stderr, "ERROR: malloc failed for sitelink_ex[%d]\n", i);
      exit(1);
    }
    memset(sitelink_ex[i], 0, V_ex*gaugeSiteSize*gSize);
  } 
  
  int nl_ghost_len[]= {
    E4*E3*E2/2 * 2, // "divided by 2" comes from even/odd division, "*4" comes from back/fwd 
    E4*E3*E1/2 * 2,
    E4*E2*E1/2 * 2,
    E3*E2*E1/2 * 2
  };
  int nl_tot_ghost_len = nl_ghost_len[0]+nl_ghost_len[1]+nl_ghost_len[2]+nl_ghost_len[3];
  
  for(i=0;i < 4;i++){
#if (CUDA_VERSION >=4000)
    cudaMallocHost((void**)&sitelink_nl[i], 2*(Vh_nl+nl_tot_ghost_len)*gaugeSiteSize* gSize);
#else
    sitelink_nl[i] = malloc(2*(Vh_nl+nl_tot_ghost_len)*gaugeSiteSize* gSize);
#endif
    if (sitelink_nl[i] == NULL){
      fprintf(stderr, "ERROR: malloc failed for sitelink_nl[%d]\n", i);
      exit(1);
    }
  } 




  //FIXME:
  //assuming all dimension size is even
  //fill in the extended sitelink 
  for(i=0; i < V_ex; i++){
      
      int sid = i;
      int oddBit=0;
      if(i >= Vh_ex){
	sid = i - Vh_ex;
	oddBit = 1;
      }

      int za = sid/E1h;
      int x1h = sid - za*E1h;
      int zb = za/E2;
      int x2 = za - zb*E2;
      int x4 = zb/E3;
      int x3 = zb - x4*E3;
      int x1odd = (x2 + x3 + x4 + oddBit) & 1;
      int x1 = 2*x1h + x1odd;      
      
      
      if( x1< 2 || x1 >= X1 +2 
	  || x2< 2 || x2 >= X2 +2 
	  || x3< 2 || x3 >= X3 +2 
	  || x4< 2 || x4 >= X4 +2){
	continue;
      }

      
      x1 = (x1 - 2 + X1) % X1;
      x2 = (x2 - 2 + X2) % X2;
      x3 = (x3 - 2 + X3) % X3;
      x4 = (x4 - 2 + X4) % X4;
      
      int idx = (x4*X3*X2*X1+x3*X2*X1+x2*X1+x1)>>1;
      if(oddBit){
	idx += Vh;
      }
      for(int dir= 0; dir < 4; dir++){
	char* src = (char*)sitelink[dir];
	char* dst = (char*)sitelink_ex[dir];	
	memcpy(dst+i*gaugeSiteSize*gSize, src+idx*gaugeSiteSize*gSize, gaugeSiteSize*gSize);	
      }//dir
  }//i
          
  gaugeParam.llfat_ga_pad = gaugeParam.ga_pad = Vsh_t;
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
  createLinkQuda(&cudaFatLink, &gaugeParam);
  
  switch(test){
  case 0:    
    {
      int Vh_2d_max = MAX(xdim*ydim/2, xdim*zdim/2);
      Vh_2d_max = MAX(Vh_2d_max, xdim*tdim/2);
      Vh_2d_max = MAX(Vh_2d_max, ydim*zdim/2);  
      Vh_2d_max = MAX(Vh_2d_max, ydim*tdim/2);  
      Vh_2d_max = MAX(Vh_2d_max, zdim*tdim/2);  
      
      gaugeParam.site_ga_pad = gaugeParam.ga_pad = 3*(Vsh_x+Vsh_y+Vsh_z+Vsh_t) + 4*Vh_2d_max;
      gaugeParam.reconstruct = link_recon;
      createLinkQuda(&cudaSiteLink, &gaugeParam);
      
      gaugeParam.staple_pad = 3*(Vsh_x + Vsh_y + Vsh_z+ Vsh_t);
      createStapleQuda(&cudaStaple, &gaugeParam);
      createStapleQuda(&cudaStaple1, &gaugeParam);
      
      break;
    }
  case 1:    
    {
      memcpy(&gaugeParam_ex, &gaugeParam, sizeof(QudaGaugeParam));
      gaugeParam_ex.X[0]= E1;
      gaugeParam_ex.X[1]= E2;
      gaugeParam_ex.X[2]= E3;
      gaugeParam_ex.X[3]= E4;
      gaugeParam_ex.site_ga_pad = gaugeParam_ex.ga_pad = E1*E2*E3/2*3;
      gaugeParam_ex.reconstruct = link_recon;
      createLinkQuda(&cudaSiteLink_ex, &gaugeParam_ex);
    
      gaugeParam_ex.staple_pad =  E1*E2*E2/2;
      createStapleQuda(&cudaStaple_ex, &gaugeParam_ex);
      createStapleQuda(&cudaStaple1_ex, &gaugeParam_ex);

      
      //set llfat_ga_gad in gaugeParam.ex as well
      gaugeParam_ex.llfat_ga_pad = gaugeParam.llfat_ga_pad;
      break;
    }
  case 2:
    {
      memcpy(&gaugeParam_nl, &gaugeParam, sizeof(QudaGaugeParam));
      gaugeParam_nl.X[0]= L1;
      gaugeParam_nl.X[1]= L2;
      gaugeParam_nl.X[2]= L3;
      gaugeParam_nl.X[3]= L4;
      gaugeParam_nl.site_ga_pad = gaugeParam_nl.ga_pad 
	= 3*(E2*E3*E4/2+ E1*E3*E4/2+E1*E2*E4/2+E1*E2*E3/2);
      gaugeParam_nl.reconstruct = link_recon;
      createLinkQuda(&cudaSiteLink_nl, &gaugeParam_nl);
      
      gaugeParam_nl.staple_pad
	= 3*(E2*E3*E4/2+ E1*E3*E4/2+E1*E2*E4/2+E1*E2*E3/2);
      createStapleQuda(&cudaStaple_nl, &gaugeParam_nl);
      createStapleQuda(&cudaStaple1_nl, &gaugeParam_nl);
      
      break;   
      
      
    }
  default:
    errorQuda("ERROR: wrong type of test in llfat\n");
  }


  initDslashConstants(cudaFatLink, 0);

    
  return;
}

static void 
llfat_end(int test)  
{  
  int i;
#if (CUDA_VERSION >= 4000)
  cudaFreeHost(fatlink);
  for(i=0;i < 4 ;i++){
    cudaFreeHost(sitelink[i]);
    cudaFreeHost(sitelink_ex[i]);
  }
#else
  free(fatlink);
  for(i=0;i < 4 ;i++){
    free(sitelink[i]);
    free(sitelink_ex[i]);
  }
  
#endif

  for(i=0;i < 4;i++){
    free(ghost_sitelink[i]);
  }
  for(i=0;i <4; i++){
    for(int j=0;j <4; j++){
      if (i==j){
	continue;
      }
      free(ghost_sitelink_diag[i*4+j]);
    }    
  }
  
  for(i=0;i < 4;i++){
    free(reflink[i]);
  }
  
  switch(test){
  case 0:
    freeLinkQuda(&cudaSiteLink);
    freeStapleQuda(&cudaStaple);
    freeStapleQuda(&cudaStaple1);
    break;
    
  case 1:
    freeLinkQuda(&cudaSiteLink_ex);
    freeStapleQuda(&cudaStaple_ex);
    freeStapleQuda(&cudaStaple1_ex);
    break;

  case 2:
    freeLinkQuda(&cudaSiteLink_nl);
    freeStapleQuda(&cudaStaple_nl);
    freeStapleQuda(&cudaStaple1_nl);
    break;

  default:
    errorQuda("Error: invalid test type\n");
  }

  freeLinkQuda(&cudaFatLink);
  exchange_llfat_cleanup();
  
  endQuda();
}



static int
llfat_test(int test) 
{
  llfat_init(test);


  float act_path_coeff_1[6];
  double act_path_coeff_2[6];
  
  for(int i=0;i < 6;i++){
    act_path_coeff_1[i]= 0.1*i;
    act_path_coeff_2[i]= 0.1*i;
  }
  
  void* act_path_coeff;    
  if(gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION){
    act_path_coeff = act_path_coeff_2;
  }else{
    act_path_coeff = act_path_coeff_1;	
  }


  //The number comes from CPU implementation in MILC, fermion_links_helpers.c    
  int flops= 61632; 

  struct timeval t0, t1, t2, t3;
  gettimeofday(&t0, NULL);

  switch(test){
  case 0:
    llfat_init_cuda(&gaugeParam);
    gaugeParam.ga_pad = gaugeParam.site_ga_pad;
    gaugeParam.reconstruct = link_recon;
    loadLinkToGPU(cudaSiteLink, sitelink, &gaugeParam);
    gettimeofday(&t1, NULL);  
    llfat_cuda(cudaFatLink, cudaSiteLink, cudaStaple, cudaStaple1, &gaugeParam, act_path_coeff_2);
    break;

  case 1:
    llfat_init_cuda_ex(&gaugeParam_ex);
    exchange_cpu_sitelink_ex(gaugeParam.X, sitelink_ex, gaugeParam.cpu_prec, 1);    
    gaugeParam_ex.ga_pad = gaugeParam_ex.site_ga_pad;
    gaugeParam_ex.reconstruct = link_recon;
    loadLinkToGPU_ex(cudaSiteLink_ex, sitelink_ex, &gaugeParam_ex);
    gettimeofday(&t1, NULL);  
    llfat_cuda_ex(cudaFatLink, cudaSiteLink_ex, cudaStaple_ex, cudaStaple1_ex, &gaugeParam, act_path_coeff_2);    
    break;

  case 2:
    llfat_init_cuda_nl(&gaugeParam_nl);
    exchange_cpu_sitelink_nl(gaugeParam.X, sitelink_ex, sitelink_nl, gaugeParam.cpu_prec, 1);    
    gaugeParam.ga_pad = gaugeParam.site_ga_pad;
    gaugeParam.reconstruct = link_recon;
    loadLinkToGPU_nl(cudaSiteLink_nl, sitelink_nl, &gaugeParam_nl);
    gettimeofday(&t1, NULL);  
    llfat_cuda_nl(cudaFatLink, cudaSiteLink_nl, cudaStaple_nl, cudaStaple1_nl, &gaugeParam, act_path_coeff_2);    

    break;
    
  default:
    errorQuda("Error:wront test type for fatlink computing\n");
  }    
  
  gettimeofday(&t2, NULL);
  storeLinkToCPU(fatlink, &cudaFatLink, &gaugeParam);
  gettimeofday(&t3, NULL);

  double secs = TDIFF(t0,t3);
 
  int i;
  void* myfatlink[4];
  for(i=0;i < 4;i++){
	myfatlink[i] = malloc(V*gaugeSiteSize*gSize);
	if(myfatlink[i] == NULL){
	  printf("Error: malloc failed for myfatlink[%d]\n", i);
	  exit(1);
	}
  }

 for(i=0;i < V; i++){
	for(int dir=0; dir< 4; dir++){
	  char* src = ((char*)fatlink)+ (4*i+dir)*gaugeSiteSize*gSize;
	  char* dst = ((char*)myfatlink[dir]) + i*gaugeSiteSize*gSize;
	  memcpy(dst, src, gaugeSiteSize*gSize);
	}
 }  

 if (verify_results){   
   int optflag = 0;
   exchange_cpu_sitelink(gaugeParam.X, sitelink, ghost_sitelink, ghost_sitelink_diag, gaugeParam.cpu_prec, optflag);
   
   
   //llfat_reference_mg(reflink, sitelink, ghost_sitelink, ghost_sitelink_diag, gaugeParam.cpu_prec, act_path_coeff);
   //llfat_reference_mg_nocomm(reflink, sitelink_ex, gaugeParam.cpu_prec, act_path_coeff);
   llfat_reference(reflink, sitelink, gaugeParam.cpu_prec, act_path_coeff);
  }
  

  int res=1;
  for(int i=0;i < 4;i++){
    res &= compare_floats(reflink[i], myfatlink[i], V*gaugeSiteSize, 1e-3, gaugeParam.cpu_prec);
  }
  int accuracy_level;
  
  accuracy_level = strong_check_link(reflink, myfatlink, V, gaugeParam.cpu_prec);  
    
  printfQuda("Test %s\n",(1 == res) ? "PASSED" : "FAILED");	    
  int volume = gaugeParam.X[0]*gaugeParam.X[1]*gaugeParam.X[2]*gaugeParam.X[3];
  double perf = 1.0* flops*volume/(secs*1024*1024*1024);
  printfQuda("gpu time =%.2f ms, flops= %.2f Gflops\n", secs*1000, perf);


  for(i=0;i < 4;i++){
	free(myfatlink[i]);
  }
  llfat_end(test);
    
  if (res == 0){//failed
    printfQuda("\n");
    printfQuda("Warning: your test failed. \n");
    printfQuda("	Did you use --verify?\n");
    printfQuda("	Did you check the GPU health by running cuda memtest?\n");
  }

  printfQuda(" h2d=%f s, computation in gpu=%f s, d2h=%f s, total time=%f s\n", 
	     TDIFF(t0, t1), TDIFF(t1, t2), TDIFF(t2, t3), TDIFF(t0, t3));
  
  return accuracy_level;
}            


static void
display_test_info(int test)
{
  printfQuda("running the following test:\n");
    
  printfQuda("link_precision           link_reconstruct           space_dimension        T_dimension       Test\n");
  printfQuda("%s                       %s                         %d/%d/%d/                  %d             %d\n", 
	     get_prec_str(prec),
	     get_recon_str(link_recon), 
	     xdim, ydim, zdim, tdim, test);
  return ;
  
}

static void
usage(char** argv )
{
  printfQuda("Usage: %s <args>\n", argv[0]);
  printfQuda("  --device <dev_id>               Set which device to run on\n");
  printfQuda("  --gprec <double/single/half>    Link precision\n"); 
  printfQuda("  --recon <8/12>                  Link reconstruction type\n"); 
  printfQuda("  --sdim <n>                      Set spacial dimention\n");
  printfQuda("  --tdim <n>                      Set T dimention size(default 24)\n"); 
  printfQuda("  --verify                        Verify the GPU results using CPU results\n");
  printfQuda("  --help                          Print out this message\n"); 
  exit(1);
  return ;
}

int 
main(int argc, char **argv) 
{

  int test = 1;
  
  //default to 18 reconstruct, 8^3 x 8 
  link_recon = QUDA_RECONSTRUCT_NO;
  xdim=ydim=zdim=tdim=8;
  cpu_prec = prec = QUDA_DOUBLE_PRECISION;

  int i;
  for (i =1;i < argc; i++){

    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }

    if( strcmp(argv[i], "--cpu_prec") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      cpu_prec =  get_prec(argv[i+1]);
      i++;
      continue;	    
    }	 
    
    if( strcmp(argv[i], "--test") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      test =  atoi(argv[i+1]);
      i++;
      continue;	    
    }
    
    if( strcmp(argv[i], "--verify") == 0){
      verify_results=1;
      continue;	    
    }	

    if( strcmp(argv[i], "--device") == 0){
      if (i+1 >= argc){
	usage(argv);
      }
      device =  atoi(argv[i+1]);
      if (device < 0){
	fprintf(stderr, "Error: invalid device number(%d)\n", device);
	exit(1);
      }
      i++;
      continue;
    }

    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  initCommsQuda(argc, argv, gridsize_from_cmdline, 4);
  
  
  display_test_info(test);

    
  int accuracy_level = llfat_test(test);
    
  printfQuda("accuracy_level=%d\n", accuracy_level);

  endCommsQuda();
  
  int ret;
  if(accuracy_level >=3 ){
    ret = 0; 
  }else{
    ret = 1; //we delclare the test failed
  }

  return ret;
}



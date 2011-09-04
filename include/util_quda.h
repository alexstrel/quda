#ifndef _UTIL_QUDA_H
#define _UTIL_QUDA_H

#include <stdio.h>
#include <stdlib.h>

#ifdef QMP_COMMS

#include <qmp.h>

#define printfQuda(...) do {	     \
 if (QMP_get_node_number() == 0) {	\
    printf(__VA_ARGS__);             \
    fflush(stdout);                  \
  }                                  \
} while (0)

#define errorQuda(...) do {		    \
  printf("QUDA error: " __VA_ARGS__);       \
  printf(" (node %d, " __FILE__ ":%d)\n",   \
         QMP_get_node_number(), __LINE__);  \
  QMP_abort(1);				    \
} while (0)

#elif defined(MPI_COMMS)
#include "mpicomm.h"
#define printfQuda(...) do {			\
    if (comm_rank() == 0) {			\
      printf(__VA_ARGS__);			\
      fflush(stdout);				\
    }						\
  } while (0)

extern char hostname[];
#define errorQuda(...) do {		    \
  printf("QUDA error: " __VA_ARGS__);       \
  printf(" (node %d, " __FILE__ ":%d, hostname=%s)\n",   \
         comm_rank(), __LINE__, hostname);			 \
  comm_exit(1);						 \
} while (0)

#else

#define printfQuda(...) do {				\
    printf(__VA_ARGS__);				\
    fflush(stdout); } while (0)				


#define errorQuda(...) do {		     \
  printf("QUDA error: " __VA_ARGS__);        \
  printf(" (" __FILE__ ":%d)\n", __LINE__);  \
  exit(1);				     \
} while (0)

#endif // USE_QMP

#define warningQuda(...) do {                \
  printfQuda("QUDA warning: " __VA_ARGS__);  \
  printfQuda("\n");                          \
} while (0)

// Only if we are host debugging will error checking be synchronous
#ifdef HOST_DEBUG
#define CUDA_DEVICE_SYNC cudaThreadSynchronize()
#else 
#define CUDA_DEVICE_SYNC 
#endif

#define checkCudaError() do {					\
    CUDA_DEVICE_SYNC;						\
    cudaError_t error = cudaGetLastError();			\
    if (error != cudaSuccess)					\
      errorQuda("(CUDA) %s", cudaGetErrorString(error));	\
  } while (0)

#ifdef __cplusplus
extern "C" {
#endif
  
  void stopwatchStart();
  double stopwatchReadSeconds();

#ifdef __cplusplus
}
#endif

#endif // _UTIL_QUDA_H

#ifndef PHYSICS_CUH
#define PHYSICS_CUH

#include <pthread.h>

extern pthread_mutex_t mutex;
extern pthread_cond_t cond;

void* physicsLoopFunc(void*);

#endif

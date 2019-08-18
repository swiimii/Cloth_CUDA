#ifndef PHYSICS_CUH
#define PHYSICS_CUH

#include <pthread.h>

#include "Particle.hpp"

extern pthread_mutex_t mutex;
extern pthread_cond_t cond;

extern bool rendering;

extern size_t particleCount;
extern Vector4* readPositions;
extern Vector4* writePositions;

extern Particle* particleBuffer;

void* physicsThreadFunc(void*);

__global__
void physicsKernel(DeviceData*);

#endif

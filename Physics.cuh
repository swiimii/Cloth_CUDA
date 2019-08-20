#ifndef PHYSICS_CUH
#define PHYSICS_CUH

#include <pthread.h>

#include "Particle.hpp"

extern pthread_mutex_t mutex;
extern pthread_cond_t cond;

extern bool rendering;

extern size_t particleCount;
extern Particle* readParticles;
extern Particle* writeParticles;

void* physicsThreadFunc(void*);

__global__
void physicsKernel(DeviceData*);

__device__
void bindingForces(Particle* particles, Vector4 (*bindingPositions)[8]);

#endif

/*
 * Physics.cuh
 *
 * Implementation files in src/Physics
 */

#ifndef PHYSICS_CUH
#define PHYSICS_CUH

#include <pthread.h>

#include <Cloth/Particle.hpp>
#include <Cloth/Helper.cuh>

// Amount of time through which the world will step each time the phsycis kernel
//   is called
#define TIME_STEP 0.001f

//----------------------------------------------------------------------------//
// Global Data
//----------------------------------------------------------------------------//

extern pthread_mutex_t mutex;
extern pthread_cond_t cond;

extern bool rendering;

extern size_t particleCount;
extern Particle* readParticles;
extern Particle* writeParticles;

//----------------------------------------------------------------------------//
// Public Function
//----------------------------------------------------------------------------//

/*
 * Thread.cu
 *
 * void* nothing
 * 	Not used. Has to be there for pthread_create.
 * return void*
 * 	Not used. Again, has to be there for pthread_create.
 *
 * Setup CUDA memory and call CUDA kernel. When main thread is done rendering,
 * 	copy device memory to host and swap host read/write buffers.
 */
void* physicsThreadFunc(void* nothing);

//----------------------------------------------------------------------------//
// Device Functions
//----------------------------------------------------------------------------//

/*
 * Kernel.cu
 *
 * DeviceData* devData
 *   Global memory pointers and information.
 *
 * Handle global/shared memory transfers and call other device kernel fucntions.
 */
__global__
void physicsKernel(DeviceData* devData, InputData inputData);

/*
 * BindingForce.cu
 *
 * Particle* particles
 *   Array of particles in shared memory.
 * Vector4 (*bindingPositions)[8]
 *   2D array of binding positions. There are eight bindings per particle, so
 *   the positions of the particles to which each particle is bound will be
 *   passed through this parameter.
 *
 * Sum binding forces. Update velocity and position vectors of each particle.
 */
__device__
void bindingForces(Particle* particles, Vector4 (*bindingPositions)[8]);

#endif

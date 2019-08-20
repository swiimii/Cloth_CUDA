#include <Cloth/Physics.cuh>

__global__
void physicsKernel(DeviceData* devData) {
	__shared__ Particle particles[4];
	__shared__ Vector4 bindingPositions[4][8];

	// BlockDim.x = 4

	size_t blockParticleIndex = threadIdx.x; // [0,4)
	size_t intraParticleIndex = threadIdx.y; // [0,8)
	size_t globalParticleIndex = (blockIdx.x << 2) + threadIdx.x;
	size_t globalBindingIndex = 0;

	//------------------------------------//
	// Read global memory
	//------------------------------------//

	// Read particles from global memory
	if(!intraParticleIndex) {
		particles[blockParticleIndex] = devData->read[globalParticleIndex];
		particles[blockParticleIndex].velocity.x[1] += -10.0 * TIME_STEP;
	}

	// Read bindings from global memory
	globalBindingIndex =
		particles[blockParticleIndex].bindings[intraParticleIndex].index;
	bindingPositions[blockParticleIndex][intraParticleIndex] =
		devData->read[globalBindingIndex].position;

	//------------------------------------//
	// Calculation
	//------------------------------------//

	bindingForces(particles, bindingPositions);

	//------------------------------------//
	// Write global memory
	//------------------------------------//

	// Update particle position and write to global memory
	if(!intraParticleIndex) {
		// write back data
		devData->write[globalParticleIndex] = particles[blockParticleIndex];
	}
}

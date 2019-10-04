#include <Cloth/Physics.cuh>

__global__
void physicsKernel(DeviceData* devData) {
	__shared__ Particle particles[4];
	__shared__ Vector4 bindingPositions[4][8];

	// BlockDim.x = 4

	int blockParticleIndex = threadIdx.x >> 3; // [0,4)
	int intraParticleIndex = threadIdx.x & 0b111; // [0,8)
	int globalParticleIndex = (blockIdx.x << 2) + blockParticleIndex;
	int globalBindingIndex = 0;

	//------------------------------------//
	// Read global memory
	//------------------------------------//

	// Read particles from global memory
	if(!intraParticleIndex) {
		particles[blockParticleIndex] = devData->read[globalParticleIndex];
		particles[blockParticleIndex].velocity.x[1] += -10.0 * TIME_STEP;
	}
//	for(int i = 0; i < 4; ++i) {
//		for(int offset = 0; offset < (sizeof(Particle) >> 2); offset += 32) {
//			if(offset + threadIdx.x < (sizeof(Particle) >> 2)) {
//				(((int*)(void*)(particles))[i])[offset + threadIdx.x] =
//					(((int*)(void*)(devData->read))[(blockIdx.x << 2) + i])[offset + threadIdx.x];
//			}
//		}
//	}

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

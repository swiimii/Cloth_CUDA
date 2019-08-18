#include "Physics.cuh"

#define TIME_STEP 0.00001f

__global__
void physicsKernel(DeviceData* devData) {
	__shared__ Particle particles[8];
	__shared__ Vector4 bindingPositions[8][8];
	__shared__ float physicalData[8][8];

	size_t blockParticleIndex = threadIdx.x; // [0,8)
	size_t intraParticleIndex = threadIdx.y; // [0,8)
	size_t globalParticleIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if(!intraParticleIndex) {
		particles[blockParticleIndex] = devData->read[globalParticleIndex];
		particles[blockParticleIndex].velocity.x[1] += -9.81 * TIME_STEP;
		particles[blockParticleIndex].position.x[1] +=
			particles[blockParticleIndex].velocity.x[1] * TIME_STEP;
		devData->write[globalParticleIndex] = particles[blockParticleIndex];
	}
}

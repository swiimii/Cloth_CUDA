#include "Physics.cuh"

#define TIME_STEP 0.00001f

__global__
void physicsKernel(DeviceData* devData) {
	__shared__ Particle particles[64];
	__shared__ Vector4 bindingPositions[64][8];
	__shared__ double physicalData[64][8];

	// BlockDim.x = 64

	size_t blockParticleIndex = threadIdx.x; // [0,64)
	size_t intraParticleIndex = threadIdx.y; // [0,8)
	size_t globalParticleIndex = (blockIdx.x << 6) + threadIdx.x;
	size_t globalBindingIndex = 0;
	size_t i, j;

	//------------------------------------//
	// Read global memory
	//------------------------------------//

	// Read particles from global memory
	if(!intraParticleIndex)
		particles[blockParticleIndex] = devData->read[globalParticleIndex];
	__syncthreads();

	// Read bindings from global memory
	globalBindingIndex =
		particles[blockParticleIndex].bindings[intraParticleIndex].index;
	bindingPositions[blockParticleIndex][intraParticleIndex] =
		devData->read[globalBindingIndex].position;

	//------------------------------------//
	// Calculation
	//------------------------------------//

	// Get distance between particles
	physicalData[blockParticleIndex][intraParticleIndex] = 0.0f;
	for(i = 0; i < 4; ++i) {
		double difference =
			bindingPositions[blockParticleIndex][intraParticleIndex].x[i]
			- particles[blockParticleIndex].position.x[i];

		bindingPositions[blockParticleIndex][intraParticleIndex].x[i] =
			difference;

		physicalData[blockParticleIndex][intraParticleIndex] +=
			difference * difference;
	}
	physicalData[blockParticleIndex][intraParticleIndex] =
		sqrt(physicalData[blockParticleIndex][intraParticleIndex])
		 + (double)!(globalParticleIndex - globalBindingIndex);
	// physicalData: the distance between particle and each binding

	// Normalize binding vectors
	for(i = 0; i < 4; ++i)
		bindingPositions[blockParticleIndex][intraParticleIndex].x[i] /=
			physicalData[blockParticleIndex][intraParticleIndex];
	// bindingPositions: the unit vector from particle to each binding

	// Get force toward a particle
	physicalData[blockParticleIndex][intraParticleIndex] -=
		particles[blockParticleIndex].bindings[intraParticleIndex].initDist;
	physicalData[blockParticleIndex][intraParticleIndex] *=
		particles[blockParticleIndex].bindings[intraParticleIndex].hooke;
	// physicalData: the force magnitude on a particle toward each binding

	// Scale binding unit vectors by forces
	for(i = 0; i < 4; ++i)
		bindingPositions[blockParticleIndex][intraParticleIndex].x[i] *=
			physicalData[blockParticleIndex][intraParticleIndex];
	// bindingPositions: force vector on particle toward each binding

	// Sum forces
	// carefully
//	size_t intraMod2 = intraParticleIndex & 1;
//	size_t intraMod4 = intraParticleIndex & 3;
//	for(i = 0; i < 2; ++i)
//		bindingPositions[blockParticleIndex][intraParticleIndex ^ intraMod2].x[i + (intraMod2<<1)] +=
//			bindingPositions[blockParticleIndex][(intraParticleIndex ^ intraMod2) + 1].x[i + (intraMod2<<1)];
//	bindingPositions[blockParticleIndex][intraMod2<<2].x[intraMod4] +=

	//------------------------------------//
	// Write global memory
	//------------------------------------//

	__syncthreads();

	// Update particle position and write to global memory
	if(!intraParticleIndex) {
		// sum forces
		for(j = 1; j < 8; ++j)
			for(i = 0; i < 4; ++i)
				bindingPositions[blockParticleIndex][0].x[i] +=
					bindingPositions[blockParticleIndex][j].x[i];
		// get acceleration
		for(i = 0; i < 4; ++i)
			bindingPositions[blockParticleIndex][0].x[i] /=
				particles[blockParticleIndex].mass;
		// update velocity from spring forces
		for(i = 0; i < 4; ++i)
			particles[blockParticleIndex].velocity.x[i] +=
				bindingPositions[blockParticleIndex][0].x[i] * TIME_STEP;
		// update velocity from gravity force
		particles[blockParticleIndex].velocity.x[1] += -9.81 * TIME_STEP;
		// update position only if it is not a fixed particle
		for(i = 0; i < 4; ++i)
			particles[blockParticleIndex].position.x[i] +=
				particles[blockParticleIndex].velocity.x[i] * TIME_STEP
				* (double)(!particles[blockParticleIndex].fixed);
		// write back data
		devData->write[globalParticleIndex] = particles[blockParticleIndex];
	}
}

#include "Physics.cuh"

#define TIME_STEP 0.001f

__global__
void physicsKernel(DeviceData* devData) {
	__shared__ Particle particles[4];
	__shared__ Vector4 bindingPositions[4][8];
	__shared__ double physicalData[4][8];

	// BlockDim.x = 4

	size_t blockParticleIndex = threadIdx.x; // [0,4)
	size_t intraParticleIndex = threadIdx.y; // [0,8)
	size_t globalParticleIndex = (blockIdx.x << 2) + threadIdx.x;
	size_t globalBindingIndex = 0;
	size_t i;

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
	particles[blockParticleIndex].bindings[intraParticleIndex].stress =
		physicalData[blockParticleIndex][intraParticleIndex]
		/ particles[blockParticleIndex].bindings[intraParticleIndex].initDist;
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
	size_t intraMod2 = intraParticleIndex & 1;
	size_t intraMod4 = intraParticleIndex & 3;
	size_t intraDiv4 = intraParticleIndex >> 2;

	bindingPositions[blockParticleIndex][intraDiv4].x[intraMod4] +=
		bindingPositions[blockParticleIndex][intraDiv4 + 4].x[intraMod4];
	bindingPositions[blockParticleIndex][intraDiv4 + 2].x[intraMod4] +=
		bindingPositions[blockParticleIndex][intraDiv4 + 6].x[intraMod4];

	bindingPositions[blockParticleIndex][intraDiv4].x[intraMod4] +=
		bindingPositions[blockParticleIndex][intraDiv4 + 2].x[intraMod4];

	if(!intraMod2) {
		i = intraParticleIndex >> 1;
		// final sum of forces into bindingPositions[0] on each particle
		bindingPositions[blockParticleIndex][0].x[i] +=
			bindingPositions[blockParticleIndex][1].x[i];
		// get acceleration
		bindingPositions[blockParticleIndex][0].x[i] /=
			particles[blockParticleIndex].mass;
		// update velocity
		particles[blockParticleIndex].velocity.x[i] +=
			bindingPositions[blockParticleIndex][0].x[i] * TIME_STEP;
		particles[blockParticleIndex].velocity.x[i] *= 0.9996;
		// update position if not fixed
		particles[blockParticleIndex].position.x[i] +=
			particles[blockParticleIndex].velocity.x[i] * TIME_STEP
			* (double)(!particles[blockParticleIndex].fixed);
	}

	//------------------------------------//
	// Write global memory
	//------------------------------------//

	// Update particle position and write to global memory
	if(!intraParticleIndex) {
		// write back data
		devData->write[globalParticleIndex] = particles[blockParticleIndex];
	}
}

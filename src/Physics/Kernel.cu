#include <Cloth/Physics.cuh>
#include <Cloth/Helper.cuh>

__global__
void physicsKernel(DeviceData* devData, InputData inputData) {
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

		// Gravity
		particles[blockParticleIndex].velocity.x[1] += -10.0 * TIME_STEP;
	}
	
	// Add force from mouse click
	if (inputData.isClicking) {
			particles[blockParticleIndex].velocity.x[1] += 
				inputData.mouseY * TIME_STEP;
			particles[blockParticleIndex].velocity.x[2] += 
				inputData.mouseZ * TIME_STEP;
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

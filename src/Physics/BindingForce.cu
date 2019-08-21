#include <Cloth/Physics.cuh>

__device__ void bindingForces(Particle* particles, Vector4 (*bindingPositions)[8]) {

	size_t blockParticleIndex = threadIdx.x; // [0,4)
	size_t intraParticleIndex = threadIdx.y; // [0,8)
	size_t globalParticleIndex = (blockIdx.x << 2) + threadIdx.x;
	size_t globalBindingIndex =
		particles[blockParticleIndex].bindings[intraParticleIndex].index;
	size_t i;

	double physicalData;
	//------------------------------------//
	// Calculation
	//------------------------------------//

	// Get distance between particles
	physicalData = 0.0f;
	for(i = 0; i < 4; ++i) {
		double difference =
			bindingPositions[blockParticleIndex][intraParticleIndex].x[i]
			- particles[blockParticleIndex].position.x[i];

		bindingPositions[blockParticleIndex][intraParticleIndex].x[i] =
			difference;

		physicalData += difference * difference;
	}
	physicalData = sqrt(physicalData)
		 + (double)!(globalParticleIndex - globalBindingIndex);
	// physicalData: the distance between particle and each binding

	// Normalize binding vectors
	for(i = 0; i < 4; ++i)
		bindingPositions[blockParticleIndex][intraParticleIndex].x[i] /=
			physicalData;
	// bindingPositions: the unit vector from particle to each binding

	// Get force toward a particle
	physicalData -=
		particles[blockParticleIndex].bindings[intraParticleIndex].initDist;
	physicalData *= (double)(physicalData > 0);
	particles[blockParticleIndex].bindings[intraParticleIndex].stress =
		physicalData
		/ particles[blockParticleIndex].bindings[intraParticleIndex].initDist;
	physicalData *=
		particles[blockParticleIndex].bindings[intraParticleIndex].hooke;
	// physicalData: the force magnitude on a particle toward each binding

	// Scale binding unit vectors by forces
	for(i = 0; i < 4; ++i)
		bindingPositions[blockParticleIndex][intraParticleIndex].x[i] *=
			physicalData;
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
		particles[blockParticleIndex].velocity.x[i] *= 0.9997;
		// update position if not fixed
		particles[blockParticleIndex].position.x[i] +=
			particles[blockParticleIndex].velocity.x[i] * TIME_STEP
			* (double)(!particles[blockParticleIndex].fixed);
	}
}

#include <pthread.h>

#include "Graphics.hpp"
#include "Physics.cuh"
#include "Input.hpp"

//----------------------------------------------------------------------------//
// Global Data
//----------------------------------------------------------------------------//

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

bool rendering = true;

size_t particleCount;
Vector4* readPositions;
Vector4* writePositions;

Particle* particleBuffer;

//----------------------------------------------------------------------------//
// Main
//----------------------------------------------------------------------------//

int main(int argc, char** argv) {

	std::vector<Particle>* tempParticles = new std::vector<Particle>;
	pthread_t physicsTID;

	std::cerr << " > Read tempParticles...\n";
	readMesh(std::cin, *tempParticles);
	std::cerr << "\t" << tempParticles->size() << "\n";

	std::cerr << " > Allocate host memory\n";
	particleCount = tempParticles->size();
	readPositions = new Vector4[particleCount];
	writePositions = new Vector4[particleCount];
	particleBuffer = new Particle[particleCount];

	std::cerr << " > Initialize host memory\n";
	for(size_t i = 0; i < particleCount; ++i) {
		particleBuffer[i] = (*tempParticles)[i];
		readPositions[i] = writePositions[i] = (*tempParticles)[i].position;
	}
	delete tempParticles;

	std::cerr << " > Begin rendering\n";

	// Start physics
	pthread_create(&physicsTID, NULL, physicsThreadFunc, particleBuffer);
	// Start graphics
	graphicsStart(&argc, argv);

	return 0;
}

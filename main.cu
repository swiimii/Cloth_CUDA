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
Particle* readParticles;
Particle* writeParticles;

//----------------------------------------------------------------------------//
// Main
//----------------------------------------------------------------------------//

int main(int argc, char** argv) {

	std::vector<Particle>* tempParticles = new std::vector<Particle>;
	pthread_t physicsTID;

	std::cerr << " > Read particles...\n";
	readMesh(std::cin, *tempParticles);
	std::cerr << "\tparticle count: " << tempParticles->size() << "\n";

	std::cerr << " > Allocate host memory\n";
	particleCount = tempParticles->size();
	readParticles = new Particle[particleCount];
	writeParticles = new Particle[particleCount];

	std::cerr << " > Initialize host memory\n";
	for(size_t i = 0; i < particleCount; ++i)
		readParticles[i] = writeParticles[i] = (*tempParticles)[i];
	delete tempParticles;

	std::cerr << " > Begin rendering\n";

	// Start physics
	pthread_create(&physicsTID, NULL, physicsThreadFunc, writeParticles);
	// Start graphics
	graphicsStart(&argc, argv);

	return 0;
}

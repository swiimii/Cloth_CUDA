#include <pthread.h>

#include "Graphics.hpp"
#include "Physics.cuh"
#include "Input.hpp"

//----------------------------------------------------------------------------//
// Global Data
//----------------------------------------------------------------------------//

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

size_t particleCount;
Vector4* readPositions;
Vector4* writePositions;

//----------------------------------------------------------------------------//
// Main
//----------------------------------------------------------------------------//

int main(int argc, char** argv) {

	std::vector<Particle> particles;
	pthread_t physicsTID;

	std::cerr << " > Read particles...\n";
	readMesh(std::cin, particles);
	std::cerr << "\t" << particles.size() << "\n";

	std::cerr << " > Allocate host memory\n";
	particleCount = particles.size();
	readPositions = new Vector4[particleCount];
	writePositions = new Vector4[particleCount];

	std::cerr << " > Initialize host memory\n";
	for(size_t i = 0; i < particleCount; ++i) {
		readPositions[i] = particles[i].position;
		std::cerr << "particle at { "
			<< particles[i].position[0] << " "
			<< particles[i].position[1] << " "
			<< particles[i].position[2] << " "
			<< particles[i].position[3] << " }\n";
	}

	std::cerr << " > Begin rendering\n";

	// Start physics
	//pthread_create(&physicsTID, NULL, physicsThreadFunc, NULL);
	// Start graphics
	graphicsStart(&argc, argv);

	return 0;
}

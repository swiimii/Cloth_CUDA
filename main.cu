#include <pthread.h>
#include <argp.h>

#include <Cloth/Graphics.hpp>
#include <Cloth/Physics.cuh>
#include <Cloth/Input.hpp>

//----------------------------------------------------------------------------//
// Global Data
//----------------------------------------------------------------------------//

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

bool rendering = true;

size_t particleCount = 0;
Particle* readParticles = nullptr;
Particle* writeParticles = nullptr;

size_t graphicsOptions = 0;

//----------------------------------------------------------------------------//
// Main
//----------------------------------------------------------------------------//

static int parse_opt(int key, char* arg, struct argp_state* state) {
	switch(key) {
	case 'b':
		graphicsOptions |= VIS_BINDINGS; break;
	case 'c':
		graphicsOptions |= VIS_COLOR; break;
	default:
		break;
	}
	return 0;
}

int main(int argc, char** argv) {

	struct argp_option options[] = {
		{ "bindings", 'b', 0, 0, "Bindings visible" },
		{ "color", 'c', 0, 0, "Enable Colors" },
		{ 0 }
	};
	struct argp argp = { options, parse_opt };
	argp_parse(&argp, argc, argv, 0, 0, 0);

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

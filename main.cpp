#include <pthread.h>

#include "Graphics.hpp"
#include "Physics.cuh"

//----------------------------------------------------------------------------//
// Global Data
//----------------------------------------------------------------------------//

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

//----------------------------------------------------------------------------//
// Main
//----------------------------------------------------------------------------//

int main(int argc, char** argv) {
	// Initialize
	myGlutInit(&argc, argv);
	// Start physics

	// Start graphics
	glutMainLoop();
	return 0;
}

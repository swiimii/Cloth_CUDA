#include "Physics.cuh"

void* physicsThreadFunc(void* nothing) {
	pthread_exit(NULL);
	return NULL;
}

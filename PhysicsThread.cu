#include "Physics.cuh"
#include <stdio.h>

void* physicsThreadFunc(void* nothing) {
	// Initialize CUDA context
	DeviceData deviceData_host = {
		.particleCount = particleCount,
		.read = NULL,
		.write = NULL
	};
	DeviceData* deviceData_dev;

	cudaMalloc(&deviceData_host.read, particleCount * sizeof(Particle));
	cudaMalloc(&deviceData_host.write, particleCount * sizeof(Particle));
	cudaMalloc(&deviceData_dev, sizeof(DeviceData));

	cudaMemcpy(
		deviceData_host.read,
		particleBuffer,
		particleCount * sizeof(Particle),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		deviceData_host.write,
		particleBuffer,
		particleCount * sizeof(Particle),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		deviceData_dev,
		&deviceData_host, 
		sizeof(DeviceData),
		cudaMemcpyHostToDevice
	);
		
physicsThreadLoop:
	for(size_t i = 0; i < 100; ++i) {
		// CUDA kernel
		physicsKernel<<<particleCount >> 3,dim3(8,8)>>>(deviceData_dev);

		// Swap device buffers
		Particle* tempParticleBuffer = deviceData_host.read;
		deviceData_host.read = deviceData_host.write;
		deviceData_host.write = tempParticleBuffer;
		cudaMemcpy(
			deviceData_dev,
			&deviceData_host, 
			sizeof(DeviceData),
			cudaMemcpyHostToDevice
		);
	}

	//fprintf(stderr,"[P] Copy dev to host\n");

	// Copy memory device to host
	cudaMemcpy(
		particleBuffer,
		deviceData_host.read,
		particleCount * sizeof(Particle),
		cudaMemcpyDeviceToHost
	);

	// Copy device buffer into host write buffer
	for(size_t i = 0; i < particleCount; ++i)
		writePositions[i] = particleBuffer[i].position;

	//if(rendering) {
	//	fprintf(stderr,"\t[P] frame dropped\n");
	//	goto physicsThreadLoop;
	//}
	while(rendering);

	// Swap host buffers
	Vector4* tempPositions = writePositions;
	writePositions = readPositions;
	readPositions = tempPositions;
	rendering = true;

	// Handoff to render thread
	goto physicsThreadLoop;
	pthread_exit(NULL);
	return NULL;
}
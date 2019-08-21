#include <Cloth/Physics.cuh>

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
		writeParticles,
		particleCount * sizeof(Particle),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		deviceData_host.write,
		writeParticles,
		particleCount * sizeof(Particle),
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		deviceData_dev,
		&deviceData_host, 
		sizeof(DeviceData),
		cudaMemcpyHostToDevice
	);
		
	const size_t minStepsPerFrame = 64;
physicsThreadLoop:
	for(size_t i = 0; i < minStepsPerFrame || rendering; ++i) {
		// CUDA kernel
		physicsKernel<<<particleCount >> 2,dim3(4,8)>>>(deviceData_dev);

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

	// Copy memory device to host
	cudaMemcpy(
		writeParticles,
		deviceData_host.read,
		particleCount * sizeof(Particle),
		cudaMemcpyDeviceToHost
	);

	// Swap host buffers
	Particle* tempParticles = writeParticles;
	writeParticles = readParticles;
	readParticles = tempParticles;
	rendering = true;

	// Handoff to render thread
	goto physicsThreadLoop;
	pthread_exit(NULL);
	return NULL;
}

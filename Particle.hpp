#ifndef PARTICLE_HPP
#define PARTICLE_HPP

struct Vector4 {
	double x[4];
	double operator[](const size_t& index) const { return this->x[index]; }
	double& operator[](const size_t& index) { return this->x[index]; }
};

struct Binding {
	int index;
	double hooke, initDist;
	float stress;
};

struct Particle {
	Vector4 position, velocity;
	double mass;
	bool fixed;
	Binding bindings[8];
};

struct DeviceData {
		size_t particleCount;
		Particle* read;
		Particle* write;
};

#endif

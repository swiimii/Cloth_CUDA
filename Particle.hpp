#ifndef PARTICLE_HPP
#define PARTICLE_HPP

struct Vector4 {
	float x[4];
	float operator[](const size_t& index) const { return this->x[index]; }
	float& operator[](const size_t& index) { return this->x[index]; }
};

struct Binding {
	int index;
	float hooke, initDist;
};

struct Particle {
	Vector4 position, velocity;
	float mass;
	bool fixed;
	Binding bindings[8];
};

struct DeviceData {
		size_t particleCount;
		Particle* read;
		Particle* write;
};

#endif

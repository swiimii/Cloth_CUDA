/*
 * Particle.hpp
 *
 * Data structures
 */

#ifndef PARTICLE_HPP
#define PARTICLE_HPP

/*
 * 4D vector of doubles
 */
struct Vector4 {
	double x[4];
	double operator[](const size_t& index) const { return this->x[index]; }
	double& operator[](const size_t& index) { return this->x[index]; }
};

/*
 * Spring between two particles
 */
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

/*
 * Global memory information for device
 */
struct DeviceData {
		size_t particleCount;
		Particle* read;
		Particle* write;
};

#endif

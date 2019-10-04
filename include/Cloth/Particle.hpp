/*
 * Particle.hpp
 *
 * Data structures
 */

#ifndef PARTICLE_HPP
#define PARTICLE_HPP

/*
 * 4D vector of floats
 */
struct Vector4 {
	float x[4];
	float operator[](const size_t& index) const { return this->x[index]; }
	float& operator[](const size_t& index) { return this->x[index]; }
};

/*
 * Spring between two particles
 */
struct Binding {
	int index;
	float hooke, initDist;
	float stress;
};

struct Particle {
	Vector4 position, velocity;
	Binding bindings[8];
	float mass;
	int fixed;
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

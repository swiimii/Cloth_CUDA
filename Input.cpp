#include "Input.hpp"

void readParticle(std::stringstream& line, std::vector<Particle>& particles);
void readBinding(std::stringstream& line, std::vector<Particle>& particles);

void readMesh(std::istream& is, std::vector<Particle>& particles) {
	std::string linestr, linetype;
	while(std::getline(is, linestr)) {
		std::stringstream liness(linestr);
		liness >> linetype;
		if(linetype == "p")
			readParticle(liness, particles);
		else if(linetype == "b")
			readBinding(liness, particles);
	}
}

/* A particle looks like this:
 *	p [fixed: 0 or 1] [position: 3 float] [mass: 1 float]
 */
void readParticle(std::stringstream& line, std::vector<Particle>& particles) {
	Particle ret;

	// Read in data
	line >> ret.fixed;
	for(int i = 0; i < 3; ++i) line >> ret.position[i];
	line >> ret.mass;

	// Initialize the rest
	ret.position[3] = 1.0;
	for(int i = 0; i < 4; ++i)
		ret.velocity[i] = 0.0f;
	for(int i = 0; i < 8; ++i) {
		ret.bindings[i].index = particles.size();
		ret.bindings[i].hooke = 0.0f;
	}

	particles.push_back(ret);
}

/* A Binding looks like this:
 * 	b [indices: 2 int] [hooke: 1 float]
 */
void readBinding(std::stringstream& line, std::vector<Particle>& particles) {
	Binding binding;
	int indices[2];
	size_t i;
	Vector4 difference;

	line >> indices[0] >> indices[1] >> binding.hooke;

	difference[0] = particles[indices[0]].position[0]
		- particles[indices[1]].position[0];
	difference[1] = particles[indices[0]].position[1]
		- particles[indices[1]].position[1];
	difference[2] = particles[indices[0]].position[2]
		- particles[indices[1]].position[2];

	binding.initDist = (float)sqrt(
		difference[0] * difference[0]
		+ difference[1] * difference[1]
		+ difference[2] * difference[2]
	);

	binding.index = indices[0];
	for(i = 0; particles[binding.index].bindings[i].index != binding.index; ++i);
	particles[binding.index].bindings[i] = binding;

	binding.index = indices[1];
	for(i = 0; particles[binding.index].bindings[i].index != binding.index; ++i);
	particles[binding.index].bindings[i] = binding;
}

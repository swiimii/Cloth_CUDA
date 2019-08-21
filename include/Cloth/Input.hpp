/*
 * Input.hpp
 *
 * Implementation files in src/Input
 */

#ifndef INPUT_HPP
#define INPUT_HPP

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>

#include "Particle.hpp"

/*
 * Input.cpp
 *
 * std::istream& is
 *   Input stream from which the scene data will be coming
 * std::vector<Particle>& particles
 *   Particle vector that will be the storage place of the incoming data
 *
 * Read in particle and binding data from an istream and store it in
 *   a vector of particles.
 */
void readMesh(std::istream&, std::vector<Particle>&);

#endif

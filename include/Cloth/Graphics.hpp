#ifndef GRAPHICS_HPP
#define GRAPHICS_HPP

#include <cmath>
#include <algorithm>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include "Particle.hpp"

#define VIS_BINDINGS 0x1
#define VIS_COLOR 0x2

extern pthread_mutex_t mutex;
extern pthread_cond_t cond;

extern bool rendering;

extern size_t particleCount;
extern Particle* readParticles;
extern Particle* writeParticles;

extern size_t graphicsOptions;

void graphicsStart(int* argc, char** argv);

void myGlutDisplayFunc();
void myGlutIdleFunc();
void myGlutKeyboardFunc(unsigned char key, int x, int y);

#endif

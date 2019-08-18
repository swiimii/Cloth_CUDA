#ifndef GRAPHICS_HPP
#define GRAPHICS_HPP

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include "Particle.hpp"

extern pthread_mutex_t mutex;
extern pthread_cond_t cond;

extern size_t particleCount;
extern Vector4* readPositions;
extern Vector4* writePositions;

void graphicsStart(int* argc, char** argv);

void myGlutDisplayFunc();
void myGlutIdleFunc();
void myGlutKeyboardFunc(unsigned char key, int x, int y);

#endif

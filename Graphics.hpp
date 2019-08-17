#ifndef GRAPHICS_HPP
#define GRAPHICS_HPP

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

extern pthread_mutex_t mutex;
extern pthread_cond_t cond;

// OpenGL
void myGlutInit(int* argc, char** argv);
void myGlutDisplayFunc();
void myGlutIdleFunc();
void myGlutKeyboardFunc(unsigned char key, int x, int y);

#endif

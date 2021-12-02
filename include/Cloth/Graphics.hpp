/*
 * Graphics.hpp
 *
 * Implementation files in src/Graphics
 */

#ifndef GRAPHICS_HPP
#define GRAPHICS_HPP

#include <cmath>
#include <algorithm>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <Cloth/Particle.hpp>

// Visibility
#define VIS_BINDINGS 0x1
#define VIS_COLOR 0x2

//----------------------------------------------------------------------------//
// Global Data
//----------------------------------------------------------------------------//

extern pthread_mutex_t mutex;
extern pthread_cond_t cond;

extern bool rendering;

extern size_t particleCount;
extern Particle* readParticles;
extern Particle* writeParticles;

extern size_t graphicsOptions;

class GraphicsState {
	public:
	static int isClicking;
	static int mouseX, mouseY;
	static int updateMouseState(int x, int y) {
		isClicking = (isClicking + 1) % 2; 
		mouseX = x;
		mouseY = y;
		return isClicking;
	};
};

//----------------------------------------------------------------------------//
// Public Function
//----------------------------------------------------------------------------//

/*
 * Graphics.cpp
 *
 * int* argc
 *   Used in glutInit. Just pass a reference to the main parameter.
 * char** argv
 *   Used in glutInit. Just pass the main parameter.
 *
 * Initialize OpenGL. Start glutMainLoop.
 */
void graphicsStart(int* argc, char** argv);

//----------------------------------------------------------------------------//
// Custom OpenGL Functions
//----------------------------------------------------------------------------//

// glutDisplayFunc -- display scene
void myGlutDisplayFunc();
// glutIdleFunc -- tell OpenGL to display scene
void myGlutIdleFunc();
// glutKeyboardFunc -- handle key press
void myGlutKeyboardFunc(unsigned char key, int x, int y);

void onMouseButtonFunction (int button, int state,  int x, int y);

void mouseMovedWhileClickedFunction(int x, int y);

#endif

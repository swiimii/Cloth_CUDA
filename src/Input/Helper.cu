#include <iostream>
#include <GL/glu.h>
#include <Cloth/Helper.cuh>

int GraphicsState::isClicking = 0;
int GraphicsState::wasClickedLastFrame = 0;
int GraphicsState::mouseY = 0;
int GraphicsState::mouseZ = 0;

// Mouse input callback
void onMouseButtonFunction (int button, int state,  int x, int y)
{
    //update that mouse is now being clicked
    GraphicsState::updateIsClicking();
    GraphicsState::wasClickedLastFrame = 1;
}

// Called when mouse is moved while a button is being clicked
void mouseMovedWhileClickedFunction(int x, int y)
{
	GraphicsState::wasClickedLastFrame = 0;
	GLint viewport[4]; //hold the viewport info
	GLdouble modelview[16]; //hold the modelview info
	GLdouble projection[16]; //hold the projection matrix info
	GLfloat winX, winY; //hold screen x,y,z coordinates
	GLdouble worldX, worldY, worldZ; //hold world x,y,z coordinates

	glGetDoublev( GL_MODELVIEW_MATRIX, modelview ); //get the modelview info
	glGetDoublev( GL_PROJECTION_MATRIX, projection ); //get the projection matrix info
	glGetIntegerv( GL_VIEWPORT, viewport ); //get the viewport info

	winX = (float)x;
	winY = (float)viewport[3] - (float)y;

	//get the world coordinates from the screen coordinates
	gluUnProject( winX, winY, 0.0, modelview, projection, viewport, &worldX, &worldY, &worldZ);
	gluUnProject( winX, winY, 1.0, modelview, projection, viewport, &worldX, &worldY, &worldZ);
	
	std::cout << "Mouse world coordinates: " << worldX << ", " << worldY << ", " << worldZ << std::endl;
	
	GraphicsState::updateMousePosition(worldY, worldZ);
}



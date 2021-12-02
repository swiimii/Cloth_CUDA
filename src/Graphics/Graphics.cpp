#include <Cloth/Graphics.hpp>
#include <GL/glu.h>
#include <iostream>

int GraphicsState::isClicking = 0;
int GraphicsState::mouseY = 0;
int GraphicsState::mouseZ = 0;

void graphicsStart(int* argc, char** argv) {
	glutInit(argc, argv);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	
	glViewport(0, 0, 800, 800);
	glutCreateWindow("ClothSimulation");
	glutDisplayFunc(myGlutDisplayFunc);
	glutIdleFunc(myGlutIdleFunc);
	glutKeyboardFunc(myGlutKeyboardFunc);
	glutMouseFunc(onMouseButtonFunction);
	glutMotionFunc(mouseMovedWhileClickedFunction);

	glutMainLoop();
}

void myGlutDisplayFunc() {
	if(!rendering) return;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	double zAxisVal = -120.0;
	gluLookAt(
	    500.0, 80.0, zAxisVal,
	    150.0, 80.0, zAxisVal,
	    0,1,0
	);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60,1.0,1.0,10000.0);

	for(int i, j = 0; j < particleCount; ++j) {
		float bindingStress, particleStress;
		particleStress = 0.0;
		glBegin(GL_LINES);
		for(i = 0; i < 8 && readParticles[j].bindings[i].index != j; ++i) {
			// Hacky, but it's just for looks
			if(graphicsOptions & VIS_COLOR) {
				bindingStress = sqrt(sqrt(std::min(1.0f,std::max(0.0f,
					readParticles[j].bindings[i].stress
				))));
				particleStress += bindingStress;

				glColor4f(bindingStress, 0.0, 1.0 - bindingStress, 1.0);
			} else {
				glColor4f(1.0, 1.0, 1.0, 1.0);
			}

			if(graphicsOptions & VIS_BINDINGS) {
				glVertex4fv(readParticles[j].position.x);
				glVertex4fv(readParticles[readParticles[j].bindings[i].index].position.x);
			}
		}
		glEnd();
		if(!(graphicsOptions & VIS_BINDINGS)) {
			glBegin(GL_POINTS);
			if(graphicsOptions & VIS_COLOR) {
				particleStress /= 8.0;
				glColor4f(particleStress, 0.0, 1.0 - particleStress, 1.0);
			}
			else glColor4f(1.0, 1.0, 1.0, 1.0);
			glVertex4fv(readParticles[j].position.x);
			glEnd();
		}
	}

	glutSwapBuffers();
	rendering = false;
}

void myGlutIdleFunc() {
	glutPostRedisplay();
}

void myGlutKeyboardFunc(unsigned char key, int x, int y) {
	switch(key) {
	case 'b':
		graphicsOptions ^= VIS_BINDINGS; break;
	case 'c':
		graphicsOptions ^= VIS_COLOR; break;
	case 27:
	case 'q':
		exit(0);
	default:
		break;
	}
}

// Mouse input callback
void onMouseButtonFunction (int button, int state,  int x, int y)
{
    //update that mouse is now being clicked
    GraphicsState::updateIsClicking();
}

// Called when mouse is moved while a button is being clicked
void mouseMovedWhileClickedFunction(int x, int y)
{
	GLint viewport[4]; //hold the viewport info
	GLdouble modelview[16]; //hold the modelview info
	GLdouble projection[16]; //hold the projection matrix info
	GLfloat winX, winY, winZ; //hold screen x,y,z coordinates
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


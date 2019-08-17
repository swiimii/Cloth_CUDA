#include "Graphics.hpp"

void myGlutInit(int* argc, char** argv) {
	glutInit(argc, argv);
	glutCreateWindow("simple");
	glutDisplayFunc(myGlutDisplayFunc);
	glutIdleFunc(myGlutIdleFunc);
	glutKeyboardFunc(myGlutKeyboardFunc);
}

void myGlutDisplayFunc() {
	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_POLYGON);
		glVertex2f(-0.5, -0.5);
		glVertex2f(-0.5, 0.5);
		glVertex2f(0.5, 0.5);
		glVertex2f(0.5, -0.5);
	glEnd();

	glFlush();
}

void myGlutIdleFunc() {
	glutPostRedisplay();
}

void myGlutKeyboardFunc(unsigned char key, int x, int y) {
	if((key == 27) | (key == 'q')) exit(0);
}

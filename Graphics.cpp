#include "Graphics.hpp"

void graphicsStart(int* argc, char** argv) {
	glutInit(argc, argv);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

	glutCreateWindow("simple");
	glutDisplayFunc(myGlutDisplayFunc);
	glutIdleFunc(myGlutIdleFunc);
	glutKeyboardFunc(myGlutKeyboardFunc);

	glutMainLoop();
}

void myGlutDisplayFunc() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	gluLookAt(
		0,0,10,
		0,0,0,
		0,1,0
	);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(170,1.0,1.0,1000.0);

	glBegin(GL_POINTS);
	glColor4f(1.0, 1.0, 1.0, 1.0);
	for(size_t i = 0; i < particleCount; ++i)
		glVertex4fv(readPositions[i].x);
	glEnd();

	glutSwapBuffers();
}

void myGlutIdleFunc() {
	glutPostRedisplay();
}

void myGlutKeyboardFunc(unsigned char key, int x, int y) {
	if((key == 27) | (key == 'q')) exit(0);
}

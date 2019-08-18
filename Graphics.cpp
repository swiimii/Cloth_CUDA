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
	if(!rendering) return;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	gluLookAt(
		-75.0, 125.0, 200.0,
		50.0, 80.0, -50.0,
		0,1,0
	);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60,1.0,1.0,1000.0);

	glBegin(GL_POINTS);
	glColor4f(1.0, 1.0, 1.0, 1.0);
	for(size_t i = 0; i < particleCount; ++i)
		glVertex4dv(readPositions[i].x);
	glEnd();

	glutSwapBuffers();
	rendering = false;
}

void myGlutIdleFunc() {
	glutPostRedisplay();
}

void myGlutKeyboardFunc(unsigned char key, int x, int y) {
	switch(key) {
	case 27:
	case 'q':
		exit(0);
	default:
		break;
	}
}

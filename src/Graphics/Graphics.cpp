#include <Cloth/Graphics.hpp>
#include <iostream>

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
	glutMouseFunc(myMouseFunction);

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
void myMouseFunction (int button, int state,  int x, int y)
{
       //getting cursor position
       std::cout << "Cursor Position at (" << x << " : " << y << ")" << std::endl;
    
}

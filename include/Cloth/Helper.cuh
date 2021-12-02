#ifndef CLOTHHELPER_CUH
#define CLOTHHELPER_CUH

class GraphicsState {
	public:
	static int isClicking;
	static int wasClickedLastFrame;
	static int mouseY, mouseZ;
	static void updateMousePosition(int y, int z) {
		mouseY = y; mouseZ = z;
	};
	static void updateIsClicking() {
		isClicking = (isClicking + 1) % 2;
	}
};

struct InputData {
public:
	int isClicking, mouseY, mouseZ;
	InputData(int ci, int myi, int mzi) : isClicking(ci), mouseY(myi), mouseZ(mzi) {}
	
};


void onMouseButtonFunction(int button, int state, int x, int y);

void mouseMovedWhileClickedFunction(int x, int y);

#endif

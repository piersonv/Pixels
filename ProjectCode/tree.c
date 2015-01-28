#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/glut.h>

void tree(float x1, float y1, float length1, float angle1, int depth, float scale, float angle2, float col)
{
	if(depth > 0)
	{
		// Draw line segment
		glColor3f(col, 0.0, 1.0-col);
		float y2 = y1 + length1 * sin(angle1);
		float x2 = x1 + length1 * cos(angle1);
		glVertex2f(x1, y1);
		glVertex2f(x2, y2);

		float new_length = length1 * scale;
		float new_angle = angle1 + angle2;
		tree(x2, y2, new_length, new_angle, depth-1, scale, angle2, col+0.075);
		new_angle = angle1 - angle2;
		tree(x2, y2, new_length, new_angle, depth-1, scale, angle2, col+0.075);
	}
}

void init()
{
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	float radius = 100;
	glOrtho(-radius, radius, -radius, radius, -radius, radius);
}

void display()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(0.0, 1.0, 1.0);
	glBegin(GL_LINES);
	tree(0, -80, 20, 1.55, 10, 0.9, 0.5, 0.1);
	glEnd();
	glFlush();
}


int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitWindowSize(520, 520);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
	glutCreateWindow("Fractal Tree CUDA");
	glutDisplayFunc(display);
	init();
	glutMainLoop();
	return 0;
}
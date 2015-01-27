//---------------------------------------
// Program: tree.cpp
// Purpose: To generate fractal trees.
// Author:  John Gauch
// Date:    Spring 2012
//---------------------------------------
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef MAC
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#define ANGLE 0.2
#define SCALE 0.9
#define RAND 0.01

//---------------------------------------
// Calculate random value between [-R..R]
//---------------------------------------
float myrand(float R)
{
   return (2 * R * rand()) / RAND_MAX - R;
}

//---------------------------------------
// Recursive function to create trees
//---------------------------------------
void tree(float x1, float y1, float length1, float angle1, int depth)
{
   if (depth > 0)
   {
      // Draw line segment
      float x2 = x1 + length1 * cos(angle1);
      float y2 = y1 + length1 * sin(angle1);
      glVertex2f(x1, y1);
      glVertex2f(x2, y2);

      // Make two recursive calls
      float length2 = length1 * (SCALE + myrand(RAND));
      float angle2 = angle1 + ANGLE + myrand(RAND);
      tree(x2, y2, length2, angle2, depth-1);
      length2 = length1 * (SCALE + myrand(RAND));
      angle2 = angle1 - ANGLE + myrand(RAND);
      tree(x2, y2, length2, angle2, depth-1);
   }
}

//---------------------------------------
// Init function for OpenGL
//---------------------------------------
void init()
{
   // Initialize OpenGL
   glClearColor(0.0, 0.0, 0.0, 1.0);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   float radius = 100;
   glOrtho(-radius, radius, -radius, radius, -radius, radius);
}

//---------------------------------------
// Display callback for OpenGL
//---------------------------------------
void display()
{
   // Draw tree
   glClear(GL_COLOR_BUFFER_BIT);
   glColor3f(0.0, 1.0, 0.0);
   glBegin(GL_LINES);
   tree(0, -80, 20, 1.5, 10);
   glEnd();
   glFlush();
}

//---------------------------------------
// Idle callback for OpenGL
//---------------------------------------
void idle()
{
   if (system("sleep 2"));
   glutPostRedisplay();
}

//---------------------------------------
// Main program
//---------------------------------------
int main(int argc, char *argv[])
{
   glutInit(&argc, argv);
   glutInitWindowSize(500, 500);
   glutInitWindowPosition(100, 100);
   glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   glutCreateWindow("Tree");
   glutDisplayFunc(display);
   glutIdleFunc(idle);
   init();
   glutMainLoop();
   return 0;
}

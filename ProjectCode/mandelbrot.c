#include <GL/gl.h>
#include <GL/glut.h>
#include <stdio.h>

#define VIEWING_DISTANCE_MIN 3.0

static GLfloat yPosition = 1.0;
static GLfloat xPosition = 1.0;
static GLfloat zPosition = 1.0;

// RGB struct for pixel colors

struct Type_rgb
{
	float r;
	float g;
	float b;
};

// Pixel variable is to contain the color values of the pixels in the picture
// Pattern variable is used to hold a pre-defined set of colors
struct Type_rgb pixels[520*520];
struct Type_rgb pattern[999];

// Mandelbrot set function
// PARALLELIZE THIS
// Finds whether or not the pixel number is in the mandelbrot set and assigns a color accordingly

void mandelbrotset()
{
	// x0 is the scaled x coordinate of the pixel (range is -2.5 to 1)
	// y0 is the scaled y coordinate of the pixel (range is -1 to 1)
	// x and y are the positions
	float x0, y0, x, y, xtemp;

	// iteration controls the number of times we iterate the set
	// max_iteration defines the maximum amount of times we can iterate
	// loc represents the location of the current x, y coordinate
	int iteration, max_iteration, loc=0;

	// Initialize all the values
	for(y0 = -1.3; y0 < 1.3; y0 += 0.005)
		for(x0 = -2.1; x0 < 0.5; x0 += 0.005)
		{
			x = 0;
			y = 0;
			iteration = 0;
			max_iteration = 1000;

			// Mandelbrot algorithm (I got it from wikipidia yay)
			while(((x*x) + (y*y) < (2*2)) && (iteration < max_iteration))
			{
				xtemp = (x*x) - (y*y) + x0;
				y = 2*x*y + y0;
				x = xtemp;
				iteration++;
			}

			if(iteration >= 999)
			{
				// If we're at the end of the iteration, set all the pixels to black
				pixels[loc].r = 0;
				pixels[loc].g = 0;
				pixels[loc].b = 0;
			}
			else
			{
				// Otherwise we need to set the color according to our pattern
				pixels[loc].r = pattern[iteration].b;
				pixels[loc].g = pattern[iteration].g;
				pixels[loc].b = pattern[iteration].r;
			}
			loc++;
		}
}


// Basic Opengl initialization
//2.6
// 520 = (-1.5 - 1.0))/0.005
void Init()
{
	glMatrixMode(GL_PROJECTION);
	glViewport(0, 0, 520, 520);

	int i;
	float r, g, b;

	// Initialize all the pixels to white
	for(i = 0; i < 520*520; i++)
	{
		pixels[i].r = 1;
		pixels[i].g = 1;
		pixels[i].b = 1;
	}

	i = 0;

	//Initialize the pattern colors up to 9^3
	for(r = 0.1; r <= 0.9; r += 0.1)
	{
		for(g = 0.1; g <= 0.9; g += 0.1)
		{
			for(b = 0.1; b <= 0.9; b += 0.1)
			{
				pattern[i].r = b;
				pattern[i].g = r;
				pattern[i].b = g;
				i++;
			}
		}
	}

	// Initialize the rest of the pattern
	for ( ; i <= 999; i++)
	{
		pattern[i].r = 1;
		pattern[i].g = 1;
		pattern[i].b = 1;
	}

	// Call the mandelbrotset function
	mandelbrotset();
}

// Basic Opengl display function
void onDisplay()
{
	// Clear the initial buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Set up viewing transformation, looking down -Z axis
	glLoadIdentity();
	//gluLookAt(0, 0, zPosition, 0, 0, -1, 0, 1, 0);
	glPixelZoom(xPosition, yPosition);

	// Draw the complete Mandelbrot set picture
	glDrawPixels(520, 520, GL_RGB, GL_FLOAT, pixels);
	//glLoadIdentity();
	glutSwapBuffers();
}

void keyPressed (unsigned char key, int x, int y)
{
	if(key == 'r')
	{
		yPosition += 0.1;
		xPosition += 0.1;
		zPosition += 0.1;
    	glutPostRedisplay();
	}
	else if(key == 'f')
	{
		xPosition -= 0.1;
		yPosition -= 0.1;
		glutPostRedisplay();
	}
}

int main(int argc, char** argv)
{
	// Basic Opengl initialization
	glutInit(&argc, argv);
	glutInitWindowSize(520, 520);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Mandelbrot Set CUDA");

	Init();

	// Connect the display function
	glutDisplayFunc(onDisplay);

	// Connect the keyboard function
	glutKeyboardFunc(keyPressed);

	// Start openGL
	glutMainLoop();
	return 0;
}

#include <GL/gl.h>
#include <GL/glut.h>
#include <stdio.h>
#include "cuda.h"
#include "book.h"

#define max_iterations 1000
#define WIDTH 520 //Size of the picture
#define blocks 40 //some derivative of WIDTH

// RGB struct for pixel colors

struct Type_rgb
{
	float r;
	float g;
	float b;
};

// Temp_pattern variable is used to hold a pre-defined set of colors
__constant__ Type_rgb temp_pattern[max_iterations-1];

Type_rgb main_pixels[WIDTH*WIDTH];
	
// Mandelbrot set function
// PARALLELIZE THIS
// Finds whether or not the pixel number is in the mandelbrot set and assigns a color accordingly

__device__ int mandelbrotset(float x0, float y0, int max_iter){
	float new_x=0, new_y=0, xsqr=0, ysqr=0;
	int iter=0; //current iteration

	//Calculate if we are within the mandelbrot set
	while((xsqr + ysqr < (2*2)) && (iter < max_iter))
			{
				new_y = 2*new_x*new_y + y0;
				new_x = xsqr - ysqr + x0; //(x*x) - (y*y) + x0
				xsqr=new_x*new_x;
				ysqr=new_y*new_y;
				iter++;
			}
	return iter;
}
	

__global__ void mandelbrotset_kernel(Type_rgb* pixels,Type_rgb* pattern,int max_iteration)
{
	float step=.005;
	int xid=blockDim.x*blockIdx.x+threadIdx.x;
	int yid=blockDim.y*blockIdx.y+threadIdx.y;
	int iteration;
	int xmin=-2.1;
	int xmax=.5;
	int ymin=-1.3;
	int ymax=1.3;
	int loc=xid;
	
	for(int j=yid+ymin; j<ymax; j+=blockDim.y*gridDim.y*step){
	    for(int i=xid+xmin; i<xmax; i+=blockDim.x*gridDim.x*step){
	    
	    iteration=mandelbrotset(i,j,max_iteration-1);

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
				pixels[loc].r = pattern[iteration].r;
				pixels[loc].g = pattern[iteration].g;
				pixels[loc].b = pattern[iteration].b;
			}
			loc+=xid;//blockDim.x*gridDim.x;
		}
	}

}

// Basic Opengl initialization
void Init()
{
	glViewport(0, 0, 520, 520);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, 520, 0, 520);

	int i;
	float r, g, b;

	Type_rgb* pattern=(Type_rgb*)malloc(sizeof(Type_rgb)*(max_iterations-1));

	// Initialize all the pixels to white
	for(i = 0; i < 520*520; i++)
	{
		main_pixels[i].r = 1;
		main_pixels[i].g = 1;
		main_pixels[i].b = 1;
	}

	i = 0;

	//Initialize the pattern colors up to 9^3
	for(r = 0.1; r <= 0.9; r += 0.1)
	{
		for(g = 0.1; g <= 0.9; g += 0.1)
		{
			for(b = 0.1; b <= 0.9; b += 0.1)
			{

				pattern[i].r = r;
				pattern[i].g = g;
				pattern[i].b = b;
				printf("%f, %f, %f\n", pattern[i].r, pattern[i].g, pattern[i].b);
				i++;
			}
		}
	}

	// Initialize the rest of the pattern
	for ( ; i <= 998; i++)
	{
		pattern[i].r = 1;
		pattern[i].g = 1;
		pattern[i].b = 1;
	}

	//Create pixel array to render picture on device
	Type_rgb* dev_pixels;

	//Capture start time
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start,0));

	//Allocate space on GPU
	HANDLE_ERROR(cudaMalloc((void**)&dev_pixels, sizeof(Type_rgb)*(WIDTH*WIDTH)));


	//Copy allocated arrays to memory
	HANDLE_ERROR(cudaMemcpyToSymbol(temp_pattern, pattern, sizeof(Type_rgb)*(max_iterations-1)));
	
	//mandelbrotset_kernel initialization
	dim3 grids(WIDTH/blocks,WIDTH,blocks);
	dim3 threads(blocks,blocks);
	mandelbrotset_kernel<<<grids,threads>>>(dev_pixels,temp_pattern,max_iterations); //begin CUDA computation

	//transfer back to pixels
	HANDLE_ERROR(cudaMemcpy(main_pixels,dev_pixels,sizeof(Type_rgb)*(WIDTH*WIDTH),cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaEventRecord(stop,0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,start,stop));
	printf("Time to generate: %3.1fms\n",elapsedTime);

	//Free memory
	free(pattern);
	HANDLE_ERROR(cudaFree(dev_pixels));

	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
}

// Basic Opengl display function
void onDisplay()
{
	// Clear the initial buffer
	glClearColor(1, 1, 1, 0);
	glClear(GL_COLOR_BUFFER_BIT);

	// Draw the complete Mandelbrot set picture
	glDrawPixels(520, 520, GL_RGB, GL_FLOAT, main_pixels);
	glutSwapBuffers();
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

	// Start openGL
	glutMainLoop();
}

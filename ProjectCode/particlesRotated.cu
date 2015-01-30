#define _USE_MATH_DEFINES
#include <cuda.h>
#include <iostream>
#include <math.h>
#include <ctime>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

// Constants
const unsigned int window_width = 768;
const unsigned int window_height = 768;

const unsigned int mesh_width = 1024;
const unsigned int mesh_height = 1024;

float rnd1[mesh_width*mesh_height];
float rnd2[mesh_width*mesh_height];

// Mouse controls
int mouse_x, mouse_y;
int buttons = 0;
float translate_z = -3.0;

// VBO variables
GLuint vbo;
void *d_vbo_buffer = NULL;

float time2 = 0.0;

// Device pointers
float4* d_array;
float* d_rnd1;
float* d_rnd2;

void keyboard(unsigned char key, int, int)
{
  	switch(key)
    {
    case(27):
      	exit(0);
      	break;
    case('a'):
      	if(buttons != 10)
  			buttons = 10;
    	else
 			buttons = 0;
    	break;
    }
}

void mouse(int button, int state, int x, int y)
{
  	if(state == GLUT_DOWN)
    {
      	buttons |= 1<<button;
    }
  	else if(state == GLUT_UP)
    {
      	buttons = 0;
    }
  	mouse_x = x;
  	mouse_y = y;
  	glutPostRedisplay();
}

void motion(int x, int y)
{
  	float dx, dy;
  	dx = x - mouse_x;
  	dy = y - mouse_y;

  	if(buttons & 4)
    {
      	translate_z += dy * 10.0;
    }
  	mouse_x = x;
  	mouse_y = y;
}

union Color
{
  float c;
  uchar4 components;
};



// Initialize the kernel
// This is done in parallel on the graphics card using the CUDA framework
/* PRESENTATION */
__global__ void initialize_kernel(float4* pos, unsigned int width, unsigned int height, float time2, float4* vel, float* rnd1, float* rnd2)
{
	/*GINGIN*/
	// Determine the x and y indices of the current thread's particle
	// What thread corresponds to what particle??? Let's find out!
	unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y+threadIdx.y;

	// Calculate the initial coordinates of position (u) and velocity (v)
    float u = x/(float)width+rnd1[y*width+x];
    float v = y/(float)height+rnd2[y*width+x];

    /*BENNY B-LIPZ*/
    // Calculate a simple sine wave pattern for the colors
    // This makes it look pretty killer
    float freq = 2.0f;
    float w = sinf(u*freq+time2)*cosf(v*freq+time2)*0.2f;

	// Set the initial color
    Color temp;
    temp.components = make_uchar4(255, 0, 255, 1);

	// Set the initial position, color, and velocity for the current particle
    pos[y*width+x] = make_float4(u, w, v, temp.c);
    vel[y*width+x] = make_float4(0.0, 0.0, 0.0, 1.0f);    
}

// Position the particles
// This is where the fun happens
__global__ void particles_kernel(float4* pos, unsigned int width, unsigned int height, float time2, float X, float Y, float4* vel, int buttons)
{
	// Speed of the particle
	const float speed = 0.0005f;

	// Limit the speed of the particle
	const float threshold = 0.1f;

	// Determine the x and y indices of the current thread's particle
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // Update position and velocity
    float u = x / (float) width;
    float v = y / (float) height;

    // Compute length away from mouse click
    // Based off change of position from particle and it's x and z distance from mouse click
    float xX = (X - width/2 + 128)/(float)width*4.5f;
    float yY = (Y - height/2 + 128)/(float)height*4.5f;
    float dx = -pos[y*width+x].x + xX;
    float dz = -pos[y*width+x].z + yY;
    float length = sqrtf(dx*dx+dz*dz);

    // Check for 'a' button
    /*BENNY B-LIPZ*/
    if(buttons == 10)
    {
    	// Add a vector pointing towards the particle's original position
    	// Will end up being a square like the mesh
  	    vel[y*width+x].x = 0;
    	vel[y*width+x].z = 0;
	    dx = -pos[y*width+x].x + u;
        dz = -pos[y*width+x].z + v;
        length = sqrtf(dx*dx+dz*dz);
        // Animate the position
        pos[y*width+x].x += dx/length*speed*10;
        pos[y*width+x].z += dz/length*speed*10;
    }

    // Check for the middle mouse button
    else if(!(buttons & 4) && !(buttons & 6))
    {
    	// Paritcles converge rapidly at target point
    	// Animate the velocity
    	float2 normalized = make_float2(dx/length*speed, dz/length*speed);
    	vel[y*width+x].x += normalized.x;
        vel[y*width+x].z += normalized.y;
        dx = vel[y*width+x].x;
        dz = vel[y*width+x].z;
        float velocity = sqrtf(dx*dx+dz*dz);

        // limit the speed of the particles
        if(velocity > threshold)
	    {
            vel[y*width+x].x = dx/velocity*threshold;
            vel[y*width+x].z = dz/velocity*threshold;
	    }

	    // Update the color
	    Color temp;
   		temp.components = make_uchar4(128/length,(int)(128/(velocity*51)),(int)(255/(velocity*51)),10);

   		// Limit where the particles can travel on each axis
   		// NEUROSCIENCE THING
      	if(pos[y*width+x].x < -5.0f && vel[y*width+x].x < 0.0)
		{
	  		vel[y*width+x].x = -vel[y*width+x].x;
		}
      	if(pos[y*width+x].x > 5.0f && vel[y*width+x].x > 0.0)
		{
	  		vel[y*width+x].x = -vel[y*width+x].x;
		}

		// Update position and color
  		pos[y*width+x].x+=vel[y*width+x].x;
   		pos[y*width+x].z+=vel[y*width+x].z;
    	pos[y*width+x].w = temp.c;
    }

    // Check for the right mouse button
    else if(!(buttons & 4))
    {
    	// Stop particles
    	vel[y*width+x].x = 0;
      vel[y*width+x].z = 0;

      // Update position and color
      pos[y*width+x].x += dx/length*speed*10;
      pos[y*width+x].z += dz/length*speed*10;
      Color temp;
      temp.components = make_uchar4(255/length, 255/2*length, 255, 10);
	    pos[y*width+x].w = temp.c;
    }

    // Make it a sine wave
	float freq = 2.0f;
    float w = sinf(u*freq + time2) * cosf(v*freq + time2) * 0.2f;

    pos[y*width+x].y=w;
}

// Process particles
void particles(GLuint vbo)
{
	//Map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    cudaGLMapBufferObject((void**)&dptr, vbo);

    //Run the particles kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
 	particles_kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, time2, mouse_x, mouse_y, d_array, buttons);

 	//Unmap buffer object
    cudaGLUnmapBufferObject(vbo);
}


// Initialize the vertex buffer object
void initialize(GLuint vbo)
{
	// Map OpenGL buffer object for writing from CUDA
	float4* dptr;
	cudaGLMapBufferObject((void**)&dptr, vbo);

	// Run the initialization kernel
	/* PRESENTATION */
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width/block.x, mesh_height/block.y, 1);
    initialize_kernel<<<grid, block>>>(dptr, mesh_width, mesh_height, time2, d_array, d_rnd1, d_rnd2);

    // Unmap the vertex buffer object
    cudaGLUnmapBufferObject(vbo);
}

// Display function
static void display(void)
{
	// Process the particles using the CUDA kernel
    particles(vbo);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // View the matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Set the rotation and translation for the camera
  	glTranslatef(0.0, 0.0, translate_z);
  	//glRotatef(90.0, 1.0, 0.0, 0.0);

  	// Render from the vertex buffer object
  	glBindBuffer(GL_ARRAY_BUFFER, vbo);

  	// Set up vertex (size, type, stride, pointer)
  	glVertexPointer(3, GL_FLOAT, 16, 0);

  	// Set up color (size, type, stride, pointer)
	glColorPointer(4, GL_UNSIGNED_BYTE, 16, (GLvoid*)12);  
	
	// Enable client-side capability
	glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    // Draw the points (mode, first, count)
    // Where count is the number of particles
    glDrawArrays(GL_POINTS, 0, mesh_width*mesh_height);

    // Disable client-side capability
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();

    // Increase the time for the wave pattern
  	time2 += 0.01;
}

// Create the vertex buffer object
void createVBO(GLuint* vbo)
{
	// Create vertex buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // Initialize VBO
    unsigned int size = mesh_width*mesh_height*4*sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register VBO with CUDA
    cudaGLRegisterBufferObject(*vbo);
}

void initGL(int argc, char** argv)
{

    glutInit(&argc, argv);

    // Setup window
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Million Particles CUDA");

    // Register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

  	//GLEW initialization
 	glewInit();

  	// Clear
  	glClearColor(0.0, 0.0, 0.0, 1.0);
  	glDisable(GL_DEPTH_TEST);

	  // Viewport
  	glViewport(0, 0, window_width, window_height);

	  // Projection
  	glMatrixMode(GL_PROJECTION);
  	glLoadIdentity();
  	gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);
}

int main(int argc, char** argv)
{
	// Initialize openGL
	initGL(argc, argv);

	// Create the vertex buffer object
	createVBO(&vbo);

	// Initialize the random arrays to be used for initial position and velocity
	for(int i=0; i<mesh_height*mesh_height; ++i)
    {
      rnd1[i] = (rand()%100-100)/2000.0f;
    }
  	for(int i=0; i<mesh_height*mesh_width; ++i)
    {
      rnd2[i] = (rand()%100-100)/2000.0f;
    }

    // CUDA allocation
    cudaMalloc(&d_array, mesh_width*mesh_height*sizeof(float4));
    cudaMalloc(&d_rnd1, mesh_width*mesh_height*sizeof(float4));
    cudaMalloc(&d_rnd2, mesh_width*mesh_height*sizeof(float));

 	//CUDA copying
    cudaMemcpy(d_rnd1, rnd1, mesh_height*mesh_width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rnd2, rnd2, mesh_height*mesh_width*sizeof(float), cudaMemcpyHostToDevice);

    // Call the initialize function
    initialize(vbo);

    glutMainLoop();
}
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

// Number of particles (mesh_width*mesh_height)
const unsigned int mesh_width = 2048;
const unsigned int mesh_height = 2048;

// Arrays with random numbers for positioning
float rnd1[mesh_width*mesh_height];
float rnd2[mesh_width*mesh_height];

// Arrays to detect chamber boundries
bool dirX[mesh_width*mesh_height];
bool dirY[mesh_width*mesh_height];

// Arrays for determining speed
float speedX[mesh_width*mesh_height];
float speedY[mesh_width*mesh_height];

// Array to detect chamber of particle
bool onRight[mesh_width*mesh_height];

// Mouse controls
int mouse_x, mouse_y;
int buttons = 0;
float translate_z = -9.0;

// VBO variables
GLuint vbo;
void *d_vbo_buffer = NULL;

float time2 = 0.0;

// Device pointers for GPU computation
float4* d_array;
float* d_rnd1;
float* d_rnd2;
bool* d_dirx;
bool* d_diry;
float* d_speedx;
float* d_speedy;
bool* d_onRight;

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
  float dy;
  dy = y - mouse_y;

  if(buttons & 4)
  {
    translate_z += dy * 0.01;
  }
  mouse_y = y;
}

union Color
{
  float c;
  uchar4 components;
};



// Initialize the kernel
// This is done in parallel on the graphics card using the CUDA framework
__global__ void initialize_kernel(float4* pos, unsigned int width, unsigned int height, float time2, float4* vel, float* rnd1, float* rnd2)
{
  // Determine the x and y indices of the current thread's particle
  unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y+threadIdx.y;

  // Calculate the initial coordinates of position (u) and velocity (v)
  float u = x/rnd1[y*width+x];
  float v = y/(float)height+rnd2[y*width+x];

  // Calculate initial color pattern
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
__global__ void particles_kernel(float4* pos, unsigned int width, unsigned int height, float time2, float4* vel, int buttons, bool* dirx, bool* diry, float* speedx, float* speedy, bool* right)
{
  // Determine the x and y indices of the current thread's particle
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

  // Setting the random speed of the particle
  float speed_x=speedx[y*width+x]*.00005;
  float speed_y=speedy[y*width+x]*.00005;

  // If particle is currently in the left chamber
  if (!right[y*width+x]){ 

    // If particle exits through the top chamber hole
    if (pos[y*width+x].x > 0.0 && pos[y*width+x].z < 4.0 && pos[y*width+x].z > 3.9){
      // Update the color
      Color temp;
      temp.components = make_uchar4(0,255,0,10);
      pos[y*width+x].w=temp.c;

      //Set this particle to right chamber
      right[y*width+x]=true;
    }

    // If particle exits through the middle chamber hole
    else if (pos[y*width+x].x > 0.0 && pos[y*width+x].z < 0.0 && pos[y*width+x].z > -0.1){
      // Update the color
      Color temp;
      temp.components = make_uchar4(0,255,0,10);
      pos[y*width+x].w=temp.c;

      //Set this particle to right chamber
      right[y*width+x]=true;
    }

    // If particle exits through the bottom chamber hole
    else if (pos[y*width+x].x > 0.0 && pos[y*width+x].z < -4.0 && pos[y*width+x].z > -4.1){
      // Update the color
      Color temp;
      temp.components = make_uchar4(0,255,0,10);
      pos[y*width+x].w=temp.c;

      // Set this particle to right chamber
      right[y*width+x]=true;
    }

    // If particle doesn't exit the left chamber
    else{
      // Keep the particle from exiting the bottom of the chamber
      if(pos[y*width+x].z < -5.0)
      {
        diry[y*width+x] = true;
      }
      // Keep the particle from exiting the top of the chamber
      else if(pos[y*width+x].z > 5.0)
      {
        diry[y*width+x] = false;
      }

      // If the particle hits the bottom of the chamber adjust speed to bounce off
      if(diry[y*width+x])
      {
        speed_y = speed_y;
      }
      // If the particle hits the top of the chamber adjust speed to bounce off
      else
      {
        speed_y = -speed_y;
      }

      // Keep the particle from exiting the left of the left chamber
      if(pos[y*width+x].x < -10.25)
      {
        dirx[y*width+x] = true;
      }
      // Keep the particle from exiting the right of the left chamber
      else if(pos[y*width+x].x > 0.0)
      {
        dirx[y*width+x] = false;
      }

      // If the particle hits the left of the chamber adjust speed to bounce off
      if(dirx[y*width+x])
      {
        speed_x = speed_x;
      }
      // If the particle hits the right of the chamber adjust speed to bounce off
      else
      {
        speed_x = -speed_x;
      }
    } // else{
  } // if (!right[y*width+x]){

  // If particle is currently in the right chamber
  else{
    // Keep the particle from exiting the bottom of the chamber
    if(pos[y*width+x].z < -5.0)
    {
      diry[y*width+x] = true;
    }
    // Keep the particle from exiting the top of the chamber
    else if(pos[y*width+x].z > 5.0)
    {
      diry[y*width+x] = false;
    }

    // If the particle hits the bottom of the chamber adjust speed to bounce off
    if(diry[y*width+x])
    {
      speed_y = speed_y;
    }
    // If the particle hits the top of the chamber adjust speed to bounce off
    else
    {
      speed_y = -speed_y;
    }

    // Keep the particle from exiting the left of the chamber
    if(pos[y*width+x].x < 0.0)
    {
      dirx[y*width+x] = true;
    }
    // Keep the particle from exiting the right of the chamber
    else if(pos[y*width+x].x > 10.25)
    {
      dirx[y*width+x] = false;
    }

    // If the particle hits the left of the chamber adjust speed to bounce off
    if(dirx[y*width+x])
    {
      speed_x = speed_x;
    }
    // If the particle hits the right of the chamber adjust speed to bounce off
    else
    {
      speed_x = -speed_x;
    }
  } // else{

    //Update particle position based on speed
    pos[y*width+x].x += speed_x;
    pos[y*width+x].z += speed_y;

}

// Process particles
void particles(GLuint vbo)
{
  // Map OpenGL buffer object for writing from CUDA
  float4 *dptr;
  cudaGLMapBufferObject((void**)&dptr, vbo);

  // Set the block and grid dimensions
  dim3 block(8, 8, 1);
  dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
  // Run the particles kernel
  particles_kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, time2, d_array, buttons, d_dirx, d_diry, d_speedx, d_speedy, d_onRight);

  //Unmap buffer object
  cudaGLUnmapBufferObject(vbo);
}


// Initialize the vertex buffer object
void initialize(GLuint vbo)
{
  // Map OpenGL buffer object for writing from CUDA
  float4* dptr;
  cudaGLMapBufferObject((void**)&dptr, vbo);

  
  // Set the block and grid dimensions
  dim3 block(8, 8, 1);
  dim3 grid(mesh_width/block.x, mesh_height/block.y, 1);
  // Run the initialize kernel
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
  glRotatef(90.0, 1.0, 0.0, 0.0);

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
  glutInitWindowSize(window_width*2, window_height);
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
  glViewport(0, 0, window_width*2, window_height);

  // Projection
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (GLfloat)window_width*2 / (GLfloat)window_height, 0.1, 10.0);
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
    rnd1[i] = (rand()%100-100)/50.0f;
  }
  for(int i=0; i<mesh_height*mesh_width; ++i)
  {
    rnd2[i] = (rand()%100-100)/50.0f;
  }

  // Initialize the arrays to detect chamber boundries
  for(int i=0; i<mesh_height*mesh_width; i++)
  {
    dirX[i] = false;
  }
  for(int i=0; i<mesh_height*mesh_width; i++)
  {
    dirY[i] = false;
  }

  // Initialize the random arrays for speed
  for(int i=0; i<mesh_height*mesh_height; ++i)
   {
      speedX[i] = rand()%500;
  }
   for(int i=0; i<mesh_height*mesh_height; ++i)
  {
    speedY[i] = rand()%500;   
  }

  // Initialize the array for determining chamber  
  for(int i=0; i<mesh_height*mesh_height; ++i)
  {
     onRight[i]=false;
  }

  // CUDA allocation
  cudaMalloc(&d_array, mesh_width*mesh_height*sizeof(float4));
  cudaMalloc(&d_rnd1, mesh_width*mesh_height*sizeof(float));
  cudaMalloc(&d_rnd2, mesh_width*mesh_height*sizeof(float));
  cudaMalloc(&d_dirx, mesh_width*mesh_height*sizeof(bool));
  cudaMalloc(&d_diry, mesh_width*mesh_height*sizeof(bool));
  cudaMalloc(&d_speedx, mesh_width*mesh_height*sizeof(float));
  cudaMalloc(&d_speedy, mesh_width*mesh_height*sizeof(float));
  cudaMalloc(&d_onRight, mesh_width*mesh_height*sizeof(bool));

  //CUDA copying
  cudaMemcpy(d_rnd1, rnd1, mesh_height*mesh_width*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rnd2, rnd2, mesh_height*mesh_width*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dirx, dirX, mesh_height*mesh_width*sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dirx, dirY, mesh_height*mesh_width*sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemcpy(d_speedx, speedX, mesh_height*mesh_width*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_speedy, speedY, mesh_height*mesh_width*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_onRight, onRight, mesh_height*mesh_width*sizeof(bool), cudaMemcpyHostToDevice);

  // Call the initialize function
  initialize(vbo);

  // Call main loop to render graphics
  glutMainLoop();
}
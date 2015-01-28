OPTS = -Wall -g -std=c++11
LIBS = -lGL -lglut -lm
ARCH := $(shell uname)
ifeq ($(ARCH), Linux)
else
 MACOSX_DEFINE = -DMACOSX -I/sw/include
 LIBS = -I/usr/common/include -I/usr/include/GL -L/System/Library/Frameworks/OpenGL.framework/Libraries -framework GLUT -framework OpenGL -lGL -lGLU -lm -lobjc -lstdc++

endif

mandelbrot: mandelbrot.o
	g++ $(OPTS) -o mandelbrot mandelbrot.o $(LIBS)

mandelbrot.o: mandelbrot.c
	g++ $(OPTS) $(MACOSX_DEFINE) -c mandelbrot.c

clean:
	rm -f *.o mandelbrot

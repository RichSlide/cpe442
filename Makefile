CC=g++
CFLAGS := $(shell pkg-config --cflags opencv4)
LIBS   := $(shell pkg-config --libs opencv4) -lpthread

all: sobel_threaded

sobel_threaded: sobel_threaded.cpp
	$(CC) $(CFLAGS) sobel_threaded.cpp -o sobel_threaded $(LIBS)

clean:
	rm -f sobel_threaded

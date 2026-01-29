CC=g++
CFLAGS := $(shell pkg-config --cflags opencv4)
LIBS   := $(shell pkg-config --libs opencv4)

all: sobel_filter.cpp
	$(CC) $(CFLAGS) sobel_filter.cpp -o sobel_filter $(LIBS)

clean:
	rm -f sobel_filer

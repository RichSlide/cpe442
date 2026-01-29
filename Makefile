CC=g++
CFLAGS := $(shell pkg-config --cflags opencv4)
LIBS   := $(shell pkg-config --libs opencv4)

sobel_filter: sobel_filter.cpp
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

clean:
	rm -f sobel_filer

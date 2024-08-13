
TARGET = mnist

CC = gcc

CFLAGS :=  
CFLAGS := $(CFLAGS) -Wall
CFLAGS := $(CFLAGS) -Wextra
CFLAGS := $(CFLAGS) -g

LDFLAGS := 
LDFLAGS := $(LDFLAGS) -lm

all: $(TARGET)

$(TARGET): main.o mnist.o bmp.o
	$(CC) $^ -o $@ $(CFLAGS) $(LDFLAGS)

main.o: main.c
	$(CC) -c $< $(CFLAGS)
	
mnist.o: mnist.c
	$(CC) -c $< $(CFLAGS)

bmp.o: ../bmp/bmp.c
	$(CC) -c $< $(CFLAGS)

.PHONY = clean
clean:
	rm *.o $(TARGET) -rf *.bmp

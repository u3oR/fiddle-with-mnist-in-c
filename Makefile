
TARGET = fiddle

CC = gcc

CFLAGS :=  
CFLAGS := $(CFLAGS) -Wall
CFLAGS := $(CFLAGS) -Wextra
CFLAGS := $(CFLAGS) -g

LDFLAGS := 
LDFLAGS := $(LDFLAGS) -lm

all: $(TARGET)

$(TARGET): main.o bmp.o mnist.o
	$(CC) $^ -o $@ $(CFLAGS) $(LDFLAGS)

main.o: main.c
	$(CC) -c $< $(CFLAGS)

bmp.o: bmp/bmp.c
	$(CC) -c $< $(CFLAGS)
	
mnist.o: mnist/mnist.c
	$(CC) -c $< $(CFLAGS)

.PHONY = clean
clean:
	rm *.o $(TARGET) -rf test_bmp test_mnist *.bmp


TARGET = bmp

CC = gcc

CFLAGS :=  
CFLAGS := $(CFLAGS) -Wall
CFLAGS := $(CFLAGS) -Wextra
CFLAGS := $(CFLAGS) -g

LDFLAGS := 
LDFLAGS := $(LDFLAGS) -lm

all: $(TARGET)

$(TARGET): main.o bmp.o
	$(CC) $^ -o $@ $(CFLAGS) $(LDFLAGS)

main.o: main.c
	$(CC) -c $< $(CFLAGS)

bmp.o: bmp.c
	$(CC) -c $< $(CFLAGS)

.PHONY = clean
clean:
	rm *.o $(TARGET) -rf *.bmp

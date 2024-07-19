
TARGET = fiddle

CC = gcc

CFLAGS := -Wall
CFLAGS := $(CFLAGS) -Wextra
CFLAGS := $(CFLAGS) -std=c11

all: $(TARGET)

$(TARGET): main.o
	$(CC) $^ -o $@ $(CFLAGS)

main.o: main.c
	$(CC) -c $< $(CFLAGS)

.PHONY = clean
clean:
	rm *.o $(TARGET) -rf

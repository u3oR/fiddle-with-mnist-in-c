
TARGET = fiddle

CC = gcc

CFLAGS :=  
CFLAGS := $(CFLAGS) -Wall
CFLAGS := $(CFLAGS) -Wextra

LDFLAGS := 
LDFLAGS := $(LDFLAGS) -lm

all: $(TARGET)

$(TARGET): main.o bmp.o
	$(CC) $^ -o $@ $(CFLAGS) $(LDFLAGS)

main.o: main.c
	$(CC) -c $< $(CFLAGS)

bmp.o: bmp.c
	$(CC) -c $< $(CFLAGS)


# test bmp.c
bmp_test: bmp.c
	$(CC) $< -o $@ $(CFLAGS) -DBMP_TEST

.PHONY = clean
clean:
	rm *.o $(TARGET) -rf bmp_test *.bmp

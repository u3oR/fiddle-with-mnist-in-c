
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

bmp.o: bmp.c
	$(CC) -c $< $(CFLAGS)
	
mnist.o: mnist.c
	$(CC) -c $< $(CFLAGS)


# test bmp.c
test_bmp: bmp.c
	$(CC) $< -o $@ $(CFLAGS) -DBMP_TEST

# test mnist.c
test_mnist: mnist.c
	$(CC) $< -o $@ $(CFLAGS) -DMNIST_TEST

.PHONY = clean
clean:
	rm *.o $(TARGET) -rf test_bmp test_mnist *.bmp

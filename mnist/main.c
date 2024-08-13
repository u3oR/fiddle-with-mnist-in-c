#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "mnist.h"

#include "../bmp/bmp.h"

int main()
{
    printf("it works \n");
    
    MnistData tMnist = MNIST_DATA_INITIALIZER;

    if(GetMnistData(&tMnist) != 0)
    {
        printf("Failed to GetMnistData in %s:%d \n", __FILE__, __LINE__);
        exit(1);
    }

    int iNum = 300;

    uint8_t *u8GrayImage = malloc(tMnist.iPixelsPerImage * sizeof(uint8_t));

    for (int i = 0; i < tMnist.iPixelsPerImage; i++)
    {
        u8GrayImage[i] = (uint8_t)(tMnist.ppdTestImages[iNum][i] * UINT8_MAX);
    }
    
    GenerateGrayBitMapFile("tmp.bmp", u8GrayImage, tMnist.iImageWidth, tMnist.iImageHeight);
    
    for (int i = 0; i < 10; i++)
    {
        printf("%lf, ", tMnist.ppdTestLabels[iNum][i]);
    }

    printf("\n");

    printf("OK to GetMnistData in %s:%d \n", __FILE__, __LINE__);

    return 0;
}

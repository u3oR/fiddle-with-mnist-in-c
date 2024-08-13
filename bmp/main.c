#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "bmp.h"


int main()
{
    int iW = 1103;
    int iH = 723;
    
    uint8_t *pcImage = malloc(sizeof(uint8_t) * iW * iH);
    
    for (int i = 0; i < iW * iH; i++)
    {
        pcImage[i] = (i % 4 == 0)?(0xFF):(0x00);
    }
    
    GenerateGrayBitMapFile("./bmp.bmp", pcImage, iW, iH);
    
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "bmp.h"

typedef struct __attribute__((packed)) {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
} BitMapFileHeader;

#define BITMAP_FILE_INITIALZIER \
        { \
            .bfType = 0x4d42, \
            .bfReserved1 = 0, \
            .bfReserved2 = 0  \
        }


typedef struct __attribute__((packed)) {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
} BitMapInfoHeader;

#define BITMAP_INFO_INITIZALIZER \
        { \
            .biSize = sizeof(BitMapInfoHeader), \
            .biCompression = 0,                 \
            .biClrUsed = 0,                     \
            .biClrImportant = 0,                \
            .biPlanes = 1,                      \
            .biXPelsPerMeter = 0,               \
            .biYPelsPerMeter = 0,               \
        }



int GenerateGrayBitMapFile(const char *strFileName, uint8_t *pcGrayImageData, int iWidth, int iHeight) 
{
    /* 设置图像的宽度和高度 */
    int iBytesPerPixel = 3; // 24位BMP图像

    /* 计算图像数据的大小 */
    int iRowSize    = (iWidth * iBytesPerPixel + 3) & (~3);
    int iImageSize  =  iRowSize * iHeight;


    /* 初始化文件头 */

    BitMapFileHeader tFileHeader = BITMAP_FILE_INITIALZIER;
    tFileHeader.bfSize           = sizeof(BitMapFileHeader) + sizeof(BitMapInfoHeader) + iImageSize;
    tFileHeader.bfOffBits        = sizeof(BitMapFileHeader) + sizeof(BitMapInfoHeader);

    BitMapInfoHeader tInfoHeader = BITMAP_INFO_INITIZALIZER;
    tInfoHeader.biWidth          = iWidth;
    tInfoHeader.biHeight         = iHeight;
    tInfoHeader.biSizeImage      = iImageSize;
    tInfoHeader.biBitCount       = 24;

    
    /* 创建图像数据缓冲区并填充内容 */

    uint8_t* pcImageData = (uint8_t*)malloc(iImageSize);

    if (pcImageData == NULL) {
        fprintf(stderr, "cannot alloc a piece of memory for storing image data\n");
        return -1;
    }

    int iRgbIndex = 0;
    int iGrayIndex = 0;

    for (int iY = 0; iY < iHeight; ++iY) 
    {
        for (int iX = 0; iX < iWidth; ++iX) 
        {
            iRgbIndex   = iY * iRowSize + iX * iBytesPerPixel;
            iGrayIndex  = iY * iWidth  + iX;

            pcImageData[iRgbIndex + 0] = pcGrayImageData[iGrayIndex];   // blue
            pcImageData[iRgbIndex + 1] = pcGrayImageData[iGrayIndex];   // green
            pcImageData[iRgbIndex + 2] = pcGrayImageData[iGrayIndex];   // red
        }
    }


    /* 将数据写入文件 */

    FILE* pFile = fopen(strFileName, "wb");

    if (pFile == NULL) 
    {
        fprintf(stderr, "cannot open file %s\n", strFileName);
        free(pcImageData);
        return -1;
    }

    // 写入文件头
    fwrite(&tFileHeader, sizeof(BitMapFileHeader), 1, pFile);
    fwrite(&tInfoHeader, sizeof(BitMapInfoHeader), 1, pFile);

    // 写入像素数据
    fwrite(pcImageData, iImageSize, 1, pFile);

    // 关闭文件
    fclose(pFile);

    // 释放内存
    free(pcImageData);

    printf("%s has been generated.\n", strFileName);

    
    return 0;
}



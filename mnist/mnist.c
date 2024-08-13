#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "mnist.h"
// #include "bmp.h"

#define MAGIC_IMAGE     2051
#define MAGIC_LABEL     2049


static void Num2Onehot(int iNum, double adOnehotLabel[10])
{
    memset(adOnehotLabel, 0, sizeof(double) * 10);

    adOnehotLabel[iNum] = 1.0f;
}


static inline void _Read4ByteBig(uint32_t *iRet, FILE *pFile)
{
    uint8_t a = 0;

    for (size_t i = 0; i < sizeof(uint32_t)/sizeof(uint8_t); i++) 
    {
        fread(&a, sizeof(uint8_t), 1, pFile);
        *iRet = (*iRet << 8) + a;
    }
}

static double **ReadImages(const char *filename)
{
    FILE *pFile = fopen(filename, "r+");
    
    /* 文件未打开 */
    if (pFile == NULL) 
    {
        printf("Cannot open file %s in Function %s\n", filename, __FUNCTION__);
        perror("fopen()->");
        exit(1);
    }

    /* 二维 */
    double **ppdData = NULL;

    uint32_t u32Magic    = 0;
    uint32_t u32ImageNum = 0;
    uint32_t u32ImageW   = 0;
    uint32_t u32ImageH   = 0;

    /* 读取魔术字 */
    _Read4ByteBig(&u32Magic   , pFile);

    /* 检查魔术字 */
    if (u32Magic != MAGIC_IMAGE)
    {
        return NULL;
    }

    /* 读取格式信息 */
    _Read4ByteBig(&u32ImageNum, pFile); /* 图像数量 */
    _Read4ByteBig(&u32ImageW  , pFile); /* 图像宽度 */
    _Read4ByteBig(&u32ImageH  , pFile); /* 图像高度 */
    
    /* 读取完格式信息，后面文件指针就开始指向数据信息了 */

    /* 第一维度存储每个二维图象的指针 */
    ppdData = malloc(sizeof(double *) * u32ImageNum);
    memset(ppdData, 0, sizeof(double *) * u32ImageNum);

    /* 临时存储像素数据用于类型转换 */
    uint8_t *pucData = malloc(sizeof(uint8_t) * u32ImageW);

    if (pucData == NULL)
    {
        perror("malloc()->");
        exit(1);
    }

    for (uint32_t i = 0; i < u32ImageNum; i++)
    {
        /* 第二维度存储图像的内容数据 */
        ppdData[i] = malloc(sizeof(double) * u32ImageW * u32ImageH);
        
        if (ppdData[i] == NULL)
        {
            perror("malloc()->");
            exit(1);
        }

        /* 从文件中读取数据并转换 */
        for (uint32_t j = 0; j < u32ImageH; j++)
        {
            /* 读取一条图像 */
            fread(pucData, sizeof(uint8_t), u32ImageW, pFile);
            
            /* 直接读取图像是上下颠倒的, 因此存储时 倒着存储 */
            /* 将字节转化成浮点值 */
            for (uint32_t k = 0; k < u32ImageW; k++)
            {
                ppdData[i][(u32ImageH - 1 - j) * u32ImageH + k] = 1.0f * pucData[k] / UINT8_MAX;
            }
            
        }
        
    }

    free(pucData);
    
    if (feof(pFile))
    {
        perror("buf ->");
        fclose(pFile);
        exit(1);
    }

    fclose(pFile);
    return ppdData;
}




static double **ReadLabels(const char *filename)
{
    FILE *pFile = fopen(filename, "r");
    
    /* 文件未打开 */
    if (pFile == NULL)
    {
        return NULL;
    }

    /*  */
    double **ppdData = NULL;

    uint32_t u32Magic    = 0;
    uint32_t u32LabelNum = 0;

    /* 读取魔术字 */
    _Read4ByteBig(&u32Magic, pFile);

    /* 检查魔术字 */
    if (u32Magic != MAGIC_LABEL)
    {
        return NULL;
    }

    _Read4ByteBig(&u32LabelNum, pFile);

    /* 标签数组 */
    ppdData = malloc(sizeof(double *) * u32LabelNum);

    uint8_t u8Pixel;

    for (uint32_t i = 0; i < u32LabelNum; i++)
    {
        ppdData[i] = malloc(sizeof(double) * 10);
        
        u8Pixel = 0;
        /* 读取一个字符 */
        fread(&u8Pixel, sizeof(uint8_t), 1, pFile);

        /* 转化成ONE-HOT编码 */
        Num2Onehot(u8Pixel, ppdData[i]);
    }
    
    if (feof(pFile))
    {
        perror("buf ->");
        fclose(pFile);
        exit(1);
    }
    
    fclose(pFile);
    return ppdData;
}



int GetMnistData(MnistData *pxMnist)
{
    /* 读取数据集文件 */

    pxMnist->ppdTrainImages = ReadImages(TRAIN_DATASET_FILE_PATH);
    if (pxMnist->ppdTrainImages == NULL) 
    {
        printf("Failed to Read Train Dataset\n");
        goto failed_to_get_data;
    }

    pxMnist->ppdTrainLabels = ReadLabels(TRAIN_LABELS_FILE_PATH);
    if (pxMnist->ppdTrainLabels == NULL) 
    {
        printf("Failed to Read Train Labels\n");
        goto failed_to_get_data;
    }

    pxMnist->ppdTestImages  = ReadImages(TEST_DATASET_FILE_PATH);
    if (pxMnist->ppdTestImages == NULL) 
    {
        printf("Failed to Read Test Dataset\n");
        goto failed_to_get_data;
    }

    pxMnist->ppdTestLabels  = ReadLabels(TEST_LABELS_FILE_PATH); 
    if (pxMnist->ppdTestLabels == NULL) 
    {
        printf("Failed to Read Test Labels\n");
        goto failed_to_get_data;
    }
    
    return 0;

failed_to_get_data:
    free(pxMnist->ppdTrainImages);
    free(pxMnist->ppdTrainLabels);
    free(pxMnist->ppdTestImages );
    free(pxMnist->ppdTestLabels );
    return -1;
}

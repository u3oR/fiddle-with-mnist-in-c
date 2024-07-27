#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "mnist.h"

#define MAGIC_IMAGE     2051
#define MAGIC_LABEL     2049


static void Num2Onehot(int iNum, double adOnehotLabel[10])
{
    memset(adOnehotLabel, 0, sizeof(double) * 10);

    adOnehotLabel[iNum] = 1.0f;
}


static inline void Read4ByteBig(uint32_t *iRet, FILE *pFile)
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
    FILE *pFile = fopen(filename, "r");
    
    /* 文件未打开 */
    if (pFile == NULL) return NULL;

    /*  */
    double **ppdData = NULL;

    uint32_t u32Magic    = 0;
    uint32_t u32ImageNum = 0;
    uint32_t u32ImageW   = 0;
    uint32_t u32ImageH   = 0;

    /* 读取魔术字 */
    Read4ByteBig(&u32Magic   , pFile);

    /* 检查魔术字 */
    if (u32Magic != MAGIC_IMAGE)
    {
        return NULL;
    }

    Read4ByteBig(&u32ImageNum, pFile);
    Read4ByteBig(&u32ImageW  , pFile);
    Read4ByteBig(&u32ImageH  , pFile);

    ppdData = malloc(sizeof(double *) * u32ImageNum);

    for (uint32_t i = 0; i < u32ImageNum; i++)
    {
        ppdData[i] = malloc(sizeof(double) * u32ImageW * u32ImageH);
        
        uint8_t u8Pixel;
        for (uint32_t j = 0; j < u32ImageW * u32ImageH; i++)
        {
            /* 读取一个字节 */
            fread(&u8Pixel, sizeof(uint8_t), 1, pFile);
            /* 将字节转化成浮点值 */
            *(ppdData[i] + j) = 1.0f * u8Pixel / UINT8_MAX;
        }
    }
    
    if (feof(pFile))
    {
        perror("buf ->");
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
    Read4ByteBig(&u32Magic, pFile);

    /* 检查魔术字 */
    if (u32Magic != MAGIC_LABEL)
    {
        return NULL;
    }

    Read4ByteBig(&u32LabelNum, pFile);

    ppdData = malloc(sizeof(double *) * u32LabelNum);

    int iPixel;
    for (uint32_t i = 0; i < u32LabelNum; i++)
    {
        ppdData[i] = malloc(sizeof(double) * 10);
        /* 读取一个字符 */
        iPixel = 0;
        fread(&iPixel, sizeof(uint8_t), 1, pFile);
        Num2Onehot(iPixel, ppdData[i]);
    }
    
    if (feof(pFile))
    {
        perror("buf ->");
        exit(1);
    }
    
    fclose(pFile);
    return ppdData;
}



int GetMnistData(MnistData *pxMnist)
{
    /* 读取数据集文件 */

    pxMnist->ppdTrainImages = ReadImages(TRAIN_DATASET_FILE_PATH);
    if (pxMnist->ppdTrainImages == NULL) {goto failed_to_get_data;}

    pxMnist->ppdTrainLabels = ReadLabels(TRAIN_LABELS_FILE_PATH);
    if (pxMnist->ppdTrainLabels == NULL) {goto failed_to_get_data;}

    pxMnist->ppdTestImages  = ReadImages(TEST_DATASET_FILE_PATH);
    if (pxMnist->ppdTestImages == NULL ) {goto failed_to_get_data;}

    pxMnist->ppdTestLabels  = ReadLabels(TEST_LABELS_FILE_PATH); 
    if (pxMnist->ppdTestLabels == NULL ) {goto failed_to_get_data;}
    
    return 0;

failed_to_get_data:
    free(pxMnist->ppdTrainImages);
    free(pxMnist->ppdTrainLabels);
    free(pxMnist->ppdTestImages );
    free(pxMnist->ppdTestLabels );
    return -1;
}




#if defined(MNIST_TEST)

int main()
{
    // FILE *pFile = fopen(TRAIN_LABELS_FILE_PATH, "r");
    // FILE *pFile = fopen(TRAIN_DATASET_FILE_PATH, "r");
    // FILE *pFile = fopen(TEST_DATASET_FILE_PATH, "r");
    FILE *pFile = fopen(TEST_LABELS_FILE_PATH, "r");

    uint32_t u32Magic = 0;
    uint32_t u32ImageNum = 0;
    uint32_t u32ImageW = 0;
    uint32_t u32ImageH = 0;

    Read4ByteBig(&u32Magic, pFile);
    Read4ByteBig(&u32ImageNum, pFile);
    Read4ByteBig(&u32ImageW, pFile);
    Read4ByteBig(&u32ImageH, pFile);

    
    printf("magic   %x\n",      u32Magic);
    printf("num     %d\n",      u32ImageNum);
    printf("W       %d\n",      u32ImageW);
    printf("H       %d\n",      u32ImageH);

    uint8_t * buf = malloc(sizeof(uint8_t) * u32ImageW * u32ImageH * u32ImageNum);

    if (buf == NULL)
    {
        perror("malloc()->");
        exit(1);
    }
    
    fread(buf, sizeof(uint8_t), u32ImageW * u32ImageH * u32ImageNum, pFile);

    if (feof(pFile))
    {
        perror("buf ->");
        exit(1);
    }
    
    fclose(pFile);
    
    return 0;
}

#endif

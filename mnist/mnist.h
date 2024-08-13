#ifndef __MNIST_H
#define __MNIST_H

#define TRAIN_DATASET_FILE_PATH "MNIST_ORG/train-images.idx3-ubyte"
#define TRAIN_LABELS_FILE_PATH  "MNIST_ORG/train-labels.idx1-ubyte"
#define TEST_DATASET_FILE_PATH  "MNIST_ORG/t10k-images.idx3-ubyte"
#define TEST_LABELS_FILE_PATH   "MNIST_ORG/t10k-labels.idx1-ubyte"

#define IMAGE_W         28
#define IMAGE_H         28

#define TRAIN_DATA_NUM  60000
#define TEST_DATA_NUM   10000

typedef struct _MnistData
{
    const int iImageWidth;
    const int iImageHeight;
    const int iPixelPerImage;
    const int iTrainNum;
    const int iTestNum;
    double **ppdTrainImages;
    double **ppdTrainLabels;
    double **ppdTestImages;
    double **ppdTestLabels;
} MnistData;

#define MNIST_DATA_INITIALIZER \
{ \
    .iImageWidth    = IMAGE_W,              \
    .iImageHeight   = IMAGE_H,              \
    .iPixelPerImage = IMAGE_W * IMAGE_H,    \
    .iTrainNum      = TRAIN_DATA_NUM,       \
    .iTestNum       = TEST_DATA_NUM,        \
}

int GetMnistData(MnistData *pxMnist);

#endif

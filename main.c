#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "mnist/mnist.h"

#ifndef __UNUESD
#define __UNUESD(x) (void)(x)
#endif

/// @brief 
/// @param pdCurtLayerOutputArray   通过激活函数后的输出
/// @param pdCurtLayerZOutputArray  当前层的原始输出
/// @param iArrayLen                原始输出层的大小
/// @return 
typedef void (*ActivationFunction) (double *pdCurtLayerOutputArray, const double *pdCurtLayerZOutputArray, int iArrayLen);

typedef void (*dActivationFunction)(double *pdCurtLayerOutputArray, const double *pdCurtLayerZOutputArray, int iArrayLen);


typedef struct _Node
{
    int     iInputSize;     /* 输入大小 */
    double *pdInputArray;   /* 输入 */

    double *pdWeightArray;  /* 权重 */
    double *pdBias;         /* 偏置 */

    double *pdZOutput;      /* 未激活输出 */
    double *pdOutput;       /* 激活输出 */
} Node;


typedef struct _Layer
{
    Node *pxNodeArray;      /* 节点列表 */
    int iNodeArraySize;     /* 该层的节点个数 */

    double *pdInputArray;   /* 输入层 */
    double *pdZOutputArray; /* 未激活输出 即 wx+b */
    double *pdOutputArray;  /* 激活的输出 */
    
    int cLayerType;
    ActivationFunction ActiveFunc;
    dActivationFunction _dActiveFunc; /* 取决于 ActiveFunc */
    void (*LayerForWard)(struct _Layer *pxLayer);
    // void (*Backward)(struct _Layer *pxLayer, double *pOutputGradient);
} Layer;

void LayerForWard(struct _Layer *pxLayer)
{
    Node *pxCurtNode = NULL;

    memset(pxLayer->pdOutputArray,  0, sizeof(double) * pxLayer->iNodeArraySize);
    memset(pxLayer->pdZOutputArray, 0, sizeof(double) * pxLayer->iNodeArraySize);
    
    /* 计算wx+b */
    for (int i = 0; i < pxLayer->iNodeArraySize; i++)
    {
        pxCurtNode = &pxLayer->pxNodeArray[i];

        for (int j = 0; j < pxCurtNode->iInputSize; j++)
        {
            pxCurtNode->pdZOutput[0] += pxCurtNode->pdInputArray[i] * pxCurtNode->pdWeightArray[i];
        }
        
        pxCurtNode->pdZOutput[0] += pxCurtNode->pdBias[0];
    }
    /* 激活函数 */
    pxLayer->ActiveFunc(pxLayer->pdOutputArray, pxLayer->pdZOutputArray, pxLayer->iNodeArraySize);

}

typedef struct _Network
{
    double *pdInputArray;       /* 网络输入 */
    int     iInputArraySize;    /* 网络输入大小 */
    double *pdOutputArray;      /* 网络输出 */
    int     iOutputArraySize;   /* 网络输出大小 */

    Layer * pxLayerArray;       /* 网络层 */
    int     iLayerArraySize;    /* 网络层个数 */

    int     _iMaxNodes;
} Network;


/// @brief 层描述表
typedef struct _LayerDeclareTable
{
    const char *sLayerType;         /* 当前层的类型 */
    int         iNodeNumber;        /* 当前层的节点数 */
    const char *sActivateFunType;   /* 当前层的激活函数类型 */
} LayerDeclareTable;

/// @brief 网络描述表
typedef struct _NetDeclareTable
{
    int                 iLayerNumber;   /* 网络层数 */
    LayerDeclareTable * pxLayerTable;   /* 网络层描述表 */
} NetDeclareTable;

void Sigmoid(double *pdCurtLayerOutputArray, const double *pdCurtLayerZOutputArray, int iArrayLen)
{
    for (int i = 0; i < iArrayLen; i++)
    {
        pdCurtLayerOutputArray[i] = (1.0f / (1.0f + exp(0.0f - pdCurtLayerZOutputArray[i])));
    }

}

void ReLU(double *pdCurtLayerOutputArray, const double *pdCurtLayerZOutputArray, int iArrayLen)
{
    double *pdOutput = pdCurtLayerOutputArray;
    const double *pdZOutput = pdCurtLayerZOutputArray;
    
    for (int i = 0; i < iArrayLen; i++)
    {
        pdOutput[i] = (pdZOutput[i] > 0) ? pdZOutput[i] : 0;
    }
}

double dReLU(double dInput)
{
    return (dInput > 0) ? (1) : (0);
}

void Softmax(double *pdCurtLayerOutputArray, const double *pdCurtLayerZOutputArray, int iArrayLen)
{
    double dSum = 0.0f;

    for (int i = 0; i < iArrayLen; i++)
    {
        pdCurtLayerOutputArray[i] = exp(pdCurtLayerZOutputArray[i]);
        dSum += pdCurtLayerOutputArray[i];
    }

    for (int i = 0; i < iArrayLen; i++)
    {
        pdCurtLayerOutputArray[i] /= dSum;
    }
}

Network *CreateAndInit(int iInputSize, NetDeclareTable *pxTable)
{
    Network *pxNet = malloc(sizeof(Network));

    /* 初始化网络 */

    pxNet->pxLayerArray = malloc(sizeof(Layer) * pxTable->iLayerNumber);
    pxNet->iLayerArraySize = pxTable->iLayerNumber;
    pxNet->iInputArraySize = iInputSize;
    pxNet->pdInputArray = malloc(sizeof(double) * iInputSize);


    Layer *pxLayer = NULL;
    LayerDeclareTable *pxLayerTable = NULL;

    for (int i = 0; i < pxNet->iLayerArraySize; i++)
    {
        /* 每一层 */
        pxLayer = pxNet->pxLayerArray + i;
        pxLayer->LayerForWard = LayerForWard;
        pxLayerTable = pxTable->pxLayerTable + i;
        
        /* 输入层(连接上一层输出) */

        // pxLayer->pdInputArray = (i == 0) ? (pxNet->pdInputArray) : ((pxLayer - 1)->pdOutputArray);

        if (i == 0) // 第一层输入新建，其他层输入连接上一层的输出
        {
            pxLayer->pdInputArray   = pxNet->pdInputArray;
        }else
        {
            pxLayer->pdInputArray   = (pxLayer - 1)->pdOutputArray;
        }

        /* 输出层 */
        pxLayer->pdZOutputArray     = malloc(sizeof(double) * pxLayerTable->iNodeNumber); /* 未激活输出层 */
        pxLayer->pdOutputArray      = malloc(sizeof(double) * pxLayerTable->iNodeNumber); /* 已激活输出层 */
        
        /* 神经元 */
        pxLayer->iNodeArraySize     = pxLayerTable->iNodeNumber;
        pxLayer->pxNodeArray        = malloc(sizeof(Node) * pxLayer->iNodeArraySize);

        for (int j = 0; j < pxLayer->iNodeArraySize; j++)
        {
            /* 连接每个节点到 */
            Node *pxNode = pxLayer->pxNodeArray + j;

            pxNode->iInputSize      = iInputSize;
            pxNode->pdInputArray    = pxLayer->pdInputArray;
            pxNode->pdZOutput       = pxLayer->pdZOutputArray + j;
            pxNode->pdOutput        = pxLayer->pdOutputArray  + j;
            pxNode->pdWeightArray   = malloc(sizeof(double) * (pxNode->iInputSize + 1));
            pxNode->pdBias          = pxNode->pdWeightArray + pxNode->iInputSize;

            /* 为权重设定随机初始值 */
            for (int k = 0; k < pxNode->iInputSize; k++)
            {
                pxNode->pdWeightArray[k] = 1.0f * rand() / RAND_MAX;
            }
            /* 为偏置设置随机初始值 */
            pxNode->pdBias[0] = 1.0f * rand() / RAND_MAX;
        }
        

        /* 激活函数 */
        if (strcmp(pxLayerTable->sActivateFunType, "ReLU") == 0)
        {
            pxLayer->ActiveFunc = ReLU;
        }
        else if (strcmp(pxLayerTable->sActivateFunType, "Sigmoid") == 0)
        {
            pxLayer->ActiveFunc = Sigmoid;
        }
        else
        {
            pxLayer->ActiveFunc = Softmax;
        }
        
        /* 下一层输入连接当前层的输出 */
        // iInputSize = pxLayer->iNodeArraySize;
        iInputSize = pxLayerTable->iNodeNumber;
    }
    
    /* 连接最后一层输出到模型的输出 */
    pxNet->iOutputArraySize = pxLayerTable->iNodeNumber;
    pxNet->pdOutputArray    = pxLayer->pdOutputArray;

    // 查找网络中节点数最多的层
    // 在反向传播时需要申请内存，以便计算足够的空间，并且避免频繁计算。
    int iMaxNodes = 0;

    for (int i = 0; i < pxNet->iLayerArraySize; i++) 
    {
        Layer *pxLayer = pxNet->pxLayerArray + i;
        
        if (pxLayer->iNodeArraySize > iMaxNodes) 
        {
            iMaxNodes = pxLayer->iNodeArraySize;
        }
    }

    pxNet->_iMaxNodes = iMaxNodes;

    return pxNet;
}


void Release(Network *pxNet)
{
    /* 每一层 */
    for (int i = 0; i < pxNet->iLayerArraySize; i++)
    {
        Layer *pxCurtLayer = pxNet->pxLayerArray + i;

        for (int j = 0; j < pxCurtLayer->iNodeArraySize; j++)
        {
            Node *pxCurtNode = pxCurtLayer->pxNodeArray + j;
            /* 节点权重/偏置 */
            free(pxCurtNode->pdWeightArray);
        }
        
        free(pxCurtLayer->pxNodeArray);
        free(pxCurtLayer->pdZOutputArray);
        free(pxCurtLayer->pdOutputArray);
    }
    
    free(pxNet->pxLayerArray);
    free(pxNet->pdInputArray);
    free(pxNet);
}


/// @brief 前向传播
/// @param pxNet 神经网络对象
/// @param pdInputArray 输入数据
void Forward(Network *pxNet, double *pdInputArray)
{
    /* 将数据复制到模型的输入层 */
    memcpy(pxNet->pdInputArray, pdInputArray, sizeof(double) * pxNet->iInputArraySize);
    /* 每一层 */
    for (int iLayerIndex = 0; iLayerIndex < pxNet->iLayerArraySize; iLayerIndex++)
    {
        Layer *pxLayer = pxNet->pxLayerArray + iLayerIndex;

        // pxLayer->LayerForWard(pxLayer);

        memset(pxLayer->pdZOutputArray, 0, sizeof(double) * pxLayer->iNodeArraySize);

        /* 每一个节点 计算累加 WX+b*/
        for (int iNodeIndex = 0; iNodeIndex < pxLayer->iNodeArraySize; iNodeIndex++)
        {
            Node *pxNode = pxLayer->pxNodeArray + iNodeIndex;
            
            

            for (int i = 0; i < pxNode->iInputSize; i++)
            {
                pxNode->pdZOutput[0] += pxNode->pdInputArray[i] * pxNode->pdWeightArray[i];
            }

            pxNode->pdZOutput[0] += pxNode->pdBias[0];

        }

        /* 经过激活函数 */
        pxLayer->ActiveFunc(pxLayer->pdOutputArray, pxLayer->pdZOutputArray, pxLayer->iNodeArraySize);
        
        // 对每个节点都单独计算激活结果
        // for (int iNodeIndex = 0; iNodeIndex < pxLayer->iNodeArraySize; iNodeIndex++)
        // {
        //     Node *pxNode = pxLayer->pxNodeArray + iNodeIndex;
        //     pxNode->pdOutput[0] = 0.0f;
        //     pxNode->pdOutput[0] = pxLayer->ActiveFunc(pxLayer->pdZOutputArray, pxLayer->iNodeArraySize, iNodeIndex);
        // }
    }
}

/// @brief 交换两个指针的地址
/// @param x 指针1的地址
/// @param y 指针2的地址
static inline void _SwapPointer(void *x, void *y)
{
    uintptr_t *px = x;
    uintptr_t *py = y;

    *px = *px ^ *py;
    *py = *px ^ *py;
    *px = *px ^ *py;
}


/// @brief 反向传播
/// @param pxNet 网络实例
/// @param dLearningRate 学习率
/// @param pdTargetOutputArray 目标输出
void Backward(Network *pxNet, double dLearningRate, double *pdTargetOutputArray)
{
    /*  */

    Layer *pxOutputLayer = pxNet->pxLayerArray + pxNet->iLayerArraySize - 1;
    
    double *pdCurtErrorArray = NULL; /* 当前层误差 */
    double *pdNextErrorArray = NULL; /* 下一层误差 */

    // 根据最大节点数分配pdErrorArray
    
    pdCurtErrorArray = malloc(pxNet->_iMaxNodes * sizeof(double));
    pdNextErrorArray = malloc(pxNet->_iMaxNodes * sizeof(double));

    memset(pdCurtErrorArray, 0, pxNet->_iMaxNodes);
    memset(pdNextErrorArray, 0, pxNet->_iMaxNodes);

    /* ===================================== */
    double dLoss = 0.0f;

    /* 从最后一层开始 */
    for (int i = pxNet->iLayerArraySize - 1; i >= 0; i--)
    {
        Layer *pxCurtLayer = pxNet->pxLayerArray + i;

        /* 输出层 */
        if (pxCurtLayer == pxOutputLayer)
        {
            /* 计算输出层误差 */
            for (int j = 0; j < pxCurtLayer->iNodeArraySize; j++)
            {
                // /* 损失函数采用 均方误差 */
                // pdCurtErrorArray[j] = pxCurtLayer->pdOutputArray[j] - pdTargetOutputArray[j];
                // dLoss += pow(pdCurtErrorArray[j], 2) / 2;

                /* 损失函数采用 交叉熵 */
                pdCurtErrorArray[j] = pxCurtLayer->pdOutputArray[j] - pdTargetOutputArray[j];
                dLoss -= log(pxCurtLayer->pdOutputArray[j]) * pdTargetOutputArray[j];
            }
            
            // printf("Loss:%.5f\n", dLoss);

            /* 更新输出层参数 */
            for (int j = 0; j < pxCurtLayer->iNodeArraySize; j++)
            {
                /* 每个节点 */
                Node *pxCurtNode = pxCurtLayer->pxNodeArray + j;
            
                /* 权重 */
                /* 输出层节点的权重个数 == 上一层的输出节点个数 */
                for (int k = 0; k < (pxCurtLayer - 1)->iNodeArraySize; k++)
                {
                    /* W_ij -= 学习率 * 误差 * 前一层输出 */
                    pxCurtNode->pdWeightArray[k] -= dLearningRate * pdCurtErrorArray[j] * (pxCurtLayer - 1)->pdOutputArray[k];
                }
                
                /* 偏置 */        
                pxCurtNode->pdBias[0] -= dLearningRate * pdCurtErrorArray[j];
            }
            
            continue;
        }
        

        /* 交换两层的误差指针 */
        _SwapPointer(&pdCurtErrorArray, &pdNextErrorArray);


        /* 隐藏层 从倒数第二层开始 */
        Layer *pxNextLayer = pxCurtLayer + 1; /* 需要下一层的参数 */
        
        /* 计算当前隐藏层误差 */
        for (int j = 0; j < pxCurtLayer->iNodeArraySize; j++)
        {
            Node *pxCurtNode = pxCurtLayer->pxNodeArray + j;
            pdCurtErrorArray[j] = 0.0f;

            for (int k = 0; k < pxNextLayer->iNodeArraySize; k++)
            {
                pdCurtErrorArray[j] += \
                    pdNextErrorArray[k] * pxNextLayer->pxNodeArray[k].pdWeightArray[j];
            }
            
            // pdCurtErrorArray[j] *= dSigmoid(pxCurtNode->pdZOutput[0]);
            pdCurtErrorArray[j] *= dReLU(pxCurtNode->pdZOutput[0]);
        }

        /* 更新隐藏层参数 */
        for (int j = 0; j < pxCurtLayer->iNodeArraySize; j++)
        {
            Node *pxCurtNode = pxCurtLayer->pxNodeArray + j;

            /* 权重 */
            if (i > 0)
            {
                for (int k = 0; k < (pxCurtLayer - 1)->iNodeArraySize; k++)
                {
                    pxCurtNode->pdWeightArray[k] -= dLearningRate * pdCurtErrorArray[j] * (pxCurtLayer - 1)->pdOutputArray[k];
                }
            }
            else /* i == 0 输入层 */
            {
                for (int k = 0; k < pxNet->iInputArraySize; k++)
                {
                    pxCurtNode->pdWeightArray[k] -= dLearningRate * pdCurtErrorArray[j] * pxNet->pdInputArray[k];
                }
            }
                        
            /* 偏置 */        
            pxCurtNode->pdBias[0] -= dLearningRate * pdCurtErrorArray[j];
        }

    }

    free(pdCurtErrorArray);
    free(pdNextErrorArray);
}


/* =============================================================================== */

#define INPUT_SIZE (784)
#define LAYER_NUM  (5)

LayerDeclareTable axLayerTable[LAYER_NUM] = {
    {.sLayerType = "Dense", .iNodeNumber = 128, .sActivateFunType = "Sigmoid"},
    {.sLayerType = "Dense", .iNodeNumber = 256, .sActivateFunType = "Sigmoid"},
    {.sLayerType = "Dense", .iNodeNumber = 128, .sActivateFunType = "Sigmoid"},
    {.sLayerType = "Dense", .iNodeNumber =  64, .sActivateFunType = "Sigmoid"},
    {.sLayerType = "Dense", .iNodeNumber =  10, .sActivateFunType = "Softmax"}
};

int main()
{
    MnistData tMnist = MNIST_DATA_INITIALIZER;
    
    if(GetMnistData(&tMnist) != 0)
    {
        printf("Get error when read dataset file.\n");
        exit(1);
    }

    srand(time(NULL));

    /* 网络结构描述 */
    NetDeclareTable xNetTable = {
        .iLayerNumber = LAYER_NUM,
        .pxLayerTable = axLayerTable    
    };

    /* 创建网络 */
    Network *pxNet = CreateAndInit(INPUT_SIZE, &xNetTable);

    // int iEpoch = 10;

    // int iBatchSize = 300;
    
    // for (int i = 0; i < iEpoch; i++)
    // {
    //     for (int j = 0; j < iBatchSize; j++)
    //     {
            
    //     }
    // }

    double *pdInput;
    double *pdOutput;

    for (int i = 0; i < tMnist.iTrainNum; i++)
    {
        // printf("Train: %d, ", i);
        pdInput  = tMnist.ppdTrainImages[i];
        pdOutput = tMnist.ppdTrainLabels[i];

        /* 前向传播 */
        Forward(pxNet, pdInput);

        /* 后向传播 */
        Backward(pxNet, 0.05, pdOutput);
        
        putchar(i % 100 == 0 ? '\n' : '/');
    }



    Release(pxNet);
    
    return 0;
}


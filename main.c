#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#ifndef __UNUESD
#define __UNUESD(x) (void)(x)
#endif


typedef double (*ActivationFunction)(double *pdArray, int iArrayLen, int iNodeIndex);

typedef struct _Node
{
    int iInputSize;
    double *pdInputArray;
    double *pdWeightArray;

    double *pdBias;
    double *pdZOutput;
    double *pdOutput;
} Node;


typedef struct _Layer
{
    Node *pxNodeArray;      /* 节点列表 */
    int iNodeArraySize;     /* 该层的节点个数 */

    double *pdInputArray;   /* 输入层 */
    double *pdZOutputArray; /* 未经过激活函数输出 即 wx+b */
    double *pdOutputArray;  /* 经过激活函数的输出 */
    
    int cLayerType;
    ActivationFunction ActiveFunc;
    // void (*ForWard)(struct _Layer *pxLayer, double *pdInputArray);
    // void (*Backward)(struct _Layer *pxLayer, double *pOutputGradient);
} Layer;



typedef struct _NetworkManager
{
    void **aAllMallocMemberArray;     // 所有动态子成员的内存地址
    int iAllMallocMemberArraySize;    // 所有动态子成员的个数
    long long llAllMallocMemorySize;  // 整个网络占据的动态空间大小
} NetworkManager;


typedef struct _Network
{
    double *pdInputArray;
    int iInputArraySize;
    double *pdOutputArray;
    int iOutputArraySize;

    Layer *pxLayerArray;
    int iLayerArraySize;

    NetworkManager xNetManager;
} Network;


typedef struct _LayerDeclareTable
{
    const char *sLayerType;
    int iNodeNumber;
    const char *sActivateFunType;
} LayerDeclareTable;

typedef struct _NetDeclareTable
{
    int iLayerNumber;
    LayerDeclareTable *pxLayerTable;
} NetDeclareTable;



double Sigmoid(double *pInput, int iInputLen, int iNodeIndex)
{
    __UNUESD(iInputLen);
    __UNUESD(iNodeIndex);
    return 1.0f / (1.0f + exp(0.0f-*pInput));
}

double ReLU(double *pInput, int iInputLen, int iNodeIndex)
{
    __UNUESD(iInputLen);
    __UNUESD(iNodeIndex);
    return (*pInput > 0) ? *pInput : 0;
}

double dReLU(double dInput)
{
    return (dInput > 0) ? (1) : (0);
}

double Softmax(double *pdArray, int iArrayLen, int iNodeIndex)
{
    double dSum = 0.0f;

    for (int i = 0; i < iArrayLen; i++)
    {
        dSum += exp(pdArray[i]);
    }
    
    return exp(pdArray[iNodeIndex]) / dSum;
}

Network *CreateAndInit(int iInputSize, NetDeclareTable *pxTable)
{
    Network *pxNet = malloc(sizeof(Network));

    /* 计算NetworkManager需要的空间 */


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
        pxLayerTable = pxTable->pxLayerTable + i;
        
        /* 输入层(连接上一层输出) */
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
            pxNode->pdWeightArray   = malloc(sizeof(double) * pxNode->iInputSize);
            pxNode->pdBias          = malloc(sizeof(double));

            /* 为权重设定随机初始值 */
            for (int k = 0; k < pxNode->iInputSize; k++)
            {
                *(pxNode->pdWeightArray + k) = 1.0f * rand() / RAND_MAX;
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

    return pxNet;
}


void Release(Network *pxNet)
{
    for (int i = 0; i < pxNet->xNetManager.iAllMallocMemberArraySize; i++)
    {
        free(pxNet->xNetManager.aAllMallocMemberArray[i]);
    }

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

        /* 每一个节点 计算累加 WX+b*/
        for (int iNodeIndex = 0; iNodeIndex < pxLayer->iNodeArraySize; iNodeIndex++)
        {
            Node *pxNode = pxLayer->pxNodeArray + iNodeIndex;
            
            pxNode->pdZOutput[0] = 0.0f;

            for (int i = 0; i < pxNode->iInputSize; i++)
            {
                pxNode->pdZOutput[0] += pxNode->pdInputArray[i] * pxNode->pdWeightArray[i];
            }

            pxNode->pdZOutput[0] += pxNode->pdBias[0];

        }

        /* 经过激活函数 */
        for (int iNodeIndex = 0; iNodeIndex < pxLayer->iNodeArraySize; iNodeIndex++)
        {
            Node *pxNode = pxLayer->pxNodeArray + iNodeIndex;

            pxNode->pdOutput[0] = 0.0f;

            pxNode->pdOutput[0] = pxLayer->ActiveFunc(pxLayer->pdZOutputArray, pxLayer->iNodeArraySize, iNodeIndex);

        }
    }
}

/// @brief 交换两个指针的地址
/// @param x 指针1的地址
/// @param y 指针2的地址
static void _SwapPointer(void *x, void *y)
{
    uintptr_t t;
    memcpy(&t, x,  sizeof(uintptr_t));
    memcpy(x,  y,  sizeof(uintptr_t));
    memcpy(y,  &t, sizeof(uintptr_t));
}

#if 0
/// @brief 反向传播
/// @param pxNet 网络实例
/// @param dLearningRate 学习率
/// @param pdTargetOutputArray 目标输出
void Backward(Network *pxNet, double dLearningRate, double *pdTargetOutputArray)
{
    Layer *pxCurtLayer = NULL;
    Layer *pxNextLayer = NULL;

    double *pdCurtErrorArray = NULL;
    double *pdNextErrorArray = NULL;
    double *pdTargetArray = NULL;
    
    pdTargetArray = pdTargetOutputArray;

    // 查找网络中节点数最多的层
    int iMaxNodes = 0;
    for (int i = 0; i < pxNet->iLayerArraySize; i++) {
        Layer *pxLayer = pxNet->pxLayerArray + i;
        if (pxLayer->iNodeArraySize > iMaxNodes) {
            iMaxNodes = pxLayer->iNodeArraySize;
        }
    }
    // 根据最大节点数分配pdErrorArray
    pdCurtErrorArray = malloc(iMaxNodes * sizeof(double));
    pdNextErrorArray = malloc(iMaxNodes * sizeof(double));

    memset(pdCurtErrorArray, 0, iMaxNodes);
    memset(pdNextErrorArray, 0, iMaxNodes);

    // 现在pdErrorArray有足够的空间存储任何一层的误差项
    // 可以在反向传播循环中使用这个数组
    
    // 最后一层输出层的误差
    Layer *pxOutputLayer = &pxNet->pxLayerArray[pxNet->iLayerArraySize - 1];

    for (int i = 0; i < pxOutputLayer->iNodeArraySize; i++)
    {
        pdCurtErrorArray[i] = pxOutputLayer->pdOutputArray[i] - pdTargetOutputArray[i];

    }
    
    for (int iLayerIndex = pxNet->iLayerArraySize - 2; iLayerIndex >= 0; iLayerIndex--)
    {
        /* 从后向前 每一层 */

        Layer *pxLayer = pxNet->pxLayerArray + iLayerIndex;
        memset(pdCurtErrorArray, 0, iMaxNodes);

        pdNextErrorArray = NULL;

        /* 计算输出层误差 */
        for (int i = 0; i < pxLayer->iNodeArraySize; i++)
        {
            /* 计算误差 */
            pdCurtErrorArray[i] = pdTargetArray[i] - pxLayer->pdOutputArray[i];
            /* 计算梯度 */
            pdCurtErrorArray[i] = pdCurtErrorArray[i] * dReLU(pxLayer->pdOutputArray[i]);
        }

        
        for (int i = 0; i < pxLayer->iNodeArraySize; i++)
        {
            /* 每个节点 */

            Node *pxNode = &(pxLayer->pxNodeArray[i]);

            /*  */
            for (int j = 0; j < pxNode->iInputSize; j++)
            {
                /* 调整权重 */
                if (iLayerIndex > 0)
                {
                    pxNode->pdWeightArray[j] -= dLearningRate * pdCurtErrorArray[j] * (pxLayer - 1)->pdOutputArray[j];
                }else
                {
                    pxNode->pdWeightArray[j] -= dLearningRate * pdCurtErrorArray[j] * pxNet->pdInputArray[j];
                }
            }

            /* 调整偏置 */
            *pxNode->pdBias -= dLearningRate * pdCurtErrorArray[i];
        }

        /* 交换两个指针的指向 */
        _SwapPointer(&pdCurtErrorArray, &pdNextErrorArray);
    }
    

    free(pdCurtErrorArray);
    free(pdNextErrorArray);

    #undef _SWAP
}

#else

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

    // 查找网络中节点数最多的层
    int iMaxNodes = 0;
    for (int i = 0; i < pxNet->iLayerArraySize; i++) {
        Layer *pxLayer = pxNet->pxLayerArray + i;
        if (pxLayer->iNodeArraySize > iMaxNodes) {
            iMaxNodes = pxLayer->iNodeArraySize;
        }
    }
    // 根据最大节点数分配pdErrorArray
    pdCurtErrorArray = malloc(iMaxNodes * sizeof(double));
    pdNextErrorArray = malloc(iMaxNodes * sizeof(double));

    memset(pdCurtErrorArray, 0, iMaxNodes);
    memset(pdNextErrorArray, 0, iMaxNodes);

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
            
            printf("Loss:%.5f\n", dLoss);

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
}

#endif

#define LAYER_NUM 3

LayerDeclareTable axLayerTable[LAYER_NUM] = {
    {.sLayerType = "Dense", .iNodeNumber = 128, .sActivateFunType = "Sigmoid"},
    {.sLayerType = "Dense", .iNodeNumber =  64, .sActivateFunType = "Sigmoid"},
    {.sLayerType = "Dense", .iNodeNumber =  10, .sActivateFunType = "Softmax"}
};

#define INPUT_SIZE (784)
double adInput[INPUT_SIZE] = {0};

int main()
{
    srand(time(NULL));

    /* 网络结构描述 */
    NetDeclareTable xNetTable = {
        .iLayerNumber = LAYER_NUM,
        .pxLayerTable = axLayerTable    
    };

    /* 创建网络 */
    Network *pxNet = CreateAndInit(INPUT_SIZE, &xNetTable);

    double adLabel[10] = {1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    /* 输出结果 */
    double dMax = 0;
    int iMaxIndex = 0;

    for (int i = 0; i < 10; i++)
    {
        printf("Train: %d, ", i);
        /* 前向传播 */
        Forward(pxNet, adInput);

        /* 输出结果向量 */
        for (int j = 0; j < pxNet->iOutputArraySize; j++)
        {
            printf("%f, ", pxNet->pdOutputArray[j]);
            if (pxNet->pdOutputArray[j] > dMax) {
                dMax = pxNet->pdOutputArray[j];
                iMaxIndex = j;
            }
        }
        printf(" Index: %d, Max: %f, ", iMaxIndex, dMax);

        /* 后向传播 */
        Backward(pxNet, 0.03, adLabel);

        printf("\n");

    }
    
    return 0;
}


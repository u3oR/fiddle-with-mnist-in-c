#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

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
    double *pdOutput;
} Node;


typedef struct _Layer
{
    Node *pxNodeArray;
    int iNodeArraySize;

    double *pdInputArray;
    double *pdOutputArray;

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
        
        /* 输入/输出 */
        if (i == 0) // 第一层输入新建，其他层输入连接上一层的输出
        {
            pxLayer->pdInputArray = pxNet->pdInputArray;
        }else
        {
            pxLayer->pdInputArray = (pxLayer - 1)->pdOutputArray;
        }

        pxLayer->pdOutputArray = malloc(sizeof(double) * pxLayerTable->iNodeNumber);

        /* 神经元 */
        pxLayer->iNodeArraySize = pxLayerTable->iNodeNumber;
        pxLayer->pxNodeArray = malloc(sizeof(Node) * pxLayer->iNodeArraySize);

        for (int j = 0; j < pxLayer->iNodeArraySize; j++)
        {
            Node *pxNode = pxLayer->pxNodeArray + j;

            pxNode->iInputSize = iInputSize;
            pxNode->pdInputArray = pxLayer->pdInputArray;
            pxNode->pdOutput = pxLayer->pdOutputArray + j;


            pxNode->pdWeightArray = malloc(sizeof(double) * pxNode->iInputSize);

            for (int k = 0; k < pxNode->iInputSize; k++)
            {
                *(pxNode->pdOutput) = 1.0f * rand() / RAND_MAX;
            }
            
            pxNode->pdBias = malloc(sizeof(double));
            
            *(pxNode->pdBias) = 1.0f * rand() / RAND_MAX;
        }
        

        /* 激活函数 */
        if (strcmp(pxLayerTable->sActivateFunType, "ReLU") == 0)
        {
            pxLayer->ActiveFunc = ReLU;
        }else
        {
            pxLayer->ActiveFunc = Softmax;
        }
        
        /* 下一层输入连接当前层的输出 */
        // iInputSize = pxLayer->iNodeArraySize;
        iInputSize = pxLayerTable->iNodeNumber;
    }
    
    /* 连接最后一层输出和模型的输出 */
    pxNet->iOutputArraySize = pxLayerTable->iNodeNumber;
    pxNet->pdOutputArray = pxLayer->pdOutputArray;


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



void Forward(Network *pxNet, double *pdInputArray)
{
    /* 将数据复制到模型的输入层 */
    memcpy(pxNet->pdInputArray, pdInputArray, sizeof(double) * pxNet->iInputArraySize);
    /* 每一层 */
    for (int iLayerIndex = 0; iLayerIndex < pxNet->iLayerArraySize; iLayerIndex++)
    {
        Layer *pxLayer = pxNet->pxLayerArray + iLayerIndex;

        /* 每一个节点 计算累加*/
        for (int iNodeIndex = 0; iNodeIndex < pxLayer->iNodeArraySize; iNodeIndex++)
        {
            Node *pxNode = pxLayer->pxNodeArray + iNodeIndex;
            
            *(pxNode->pdOutput) = 0.0f;

            for (int i = 0; i < pxNode->iInputSize; i++)
            {
                *(pxNode->pdOutput) += pxNode->pdInputArray[i] * pxNode->pdWeightArray[i];
            }

            *(pxNode->pdOutput) += *(pxNode->pdBias);

        }

        /* 激活函数 */
        for (int iNodeIndex = 0; iNodeIndex < pxLayer->iNodeArraySize; iNodeIndex++)
        {
            Node *pxNode = pxLayer->pxNodeArray + iNodeIndex;

            for (int i = 0; i < pxNode->iInputSize; i++)
            {
                /* 通过激活函数 */
                *(pxNode->pdOutput) = pxLayer->ActiveFunc(pxLayer->pdOutputArray, pxLayer->iNodeArraySize, i);
            }
        }
    }
}


void Backward(Network *pxNet, double dLearningRate, double *pdTargetOutputArray)
{
    double *pdErrorArray = NULL;
    double *pdTargetArray = NULL;
    
    pdTargetArray = pdTargetOutputArray;

    for (int iLayerIndex = pxNet->iLayerArraySize - 1; iLayerIndex >= 0; iLayerIndex--)
    {
        /* 从后向前 每一层 */

        Layer *pxLayer = pxNet->pxLayerArray + iLayerIndex;
        pdErrorArray = malloc(sizeof(double) * pxLayer->iNodeArraySize);

        for (int i = 0; i < pxLayer->iNodeArraySize; i++)
        {
            /* 计算误差 */
            pdErrorArray[i] = pdTargetArray[i] - pxLayer->pdOutputArray[i];
            /* 计算梯度 */
            pdErrorArray[i] = pdErrorArray[i] * dReLU(pxLayer->pdOutputArray[i]);
        }

        
        for (int i = 0; i < pxLayer->iNodeArraySize; i++)
        {
            /* 每个节点 */

            Node *pxNode = &(pxLayer->pxNodeArray[i]);

            /*  */
            for (int j = 0; j < pxNode->iInputSize; j++)
            {
                /* 调整权重 */
                // pxNode->pdWeightArray[j] -= dLearningRate *;
            }

            /* 调整偏置 */
            // *pxNode->pdBias -= dLearningRate * ;
        }

    }
    
}


#define LAYER_NUM 3

LayerDeclareTable axLayerTable[LAYER_NUM] = {
    {.sLayerType = "Dense", .iNodeNumber = 128, .sActivateFunType = "ReLU"},
    {.sLayerType = "Dense", .iNodeNumber = 64, .sActivateFunType = "ReLU"},
    {.sLayerType = "Dense", .iNodeNumber = 10, .sActivateFunType = "Softmax"}
};

#define INPUT_SIZE (784)
double adInput[INPUT_SIZE] = {0};

int main()
{
    /* 网络结构描述 */
    NetDeclareTable xNetTable = {
        .iLayerNumber = LAYER_NUM,
        .pxLayerTable = axLayerTable    
    };

    /* 创建网络 */
    Network *pxNet = CreateAndInit(INPUT_SIZE, &xNetTable);

    /* 前向传播 */
    Forward(pxNet, adInput);
    
    /* 输出结果 */

    double dSum = 0;
    
    for (int i = 0; i < pxNet->iOutputArraySize; i++)
    {
        dSum += pxNet->pdOutputArray[i];
        
    }
    printf("dSum = %f\n", dSum);

    double dSum2 = 0;
    for (int i = 0; i < pxNet->iOutputArraySize; i++)
    {
        dSum2 += pxNet->pdOutputArray[i] / dSum;
        printf("%f, ", pxNet->pdOutputArray[i] / dSum);
    }
    printf("\n");
    printf("dSum2 = %f\n", dSum2);

    return 0;
}


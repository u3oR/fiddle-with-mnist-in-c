#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>



typedef struct _Node
{
    int iInputSize;
    double *pdInput;
    double *pdWeight;

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
    double (*Activate)(double);
    void (*ForWard)(struct _Layer *pxLayer, double *pdInput);
    void (*Backward)(struct _Layer *pxLayer, double *pOutputGradient);
} Layer;



typedef struct _NetworkManager
{
    void **aAllMallocMemberArray;     // 所有动态子成员的内存地址
    int iAllMallocMemberArraySize;    // 所有动态子成员的个数
    long long llAllMallocMemorySize;  // 整个网络占据的动态空间大小
} NetworkManager;


typedef struct _Network
{
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


inline double Sigmoid(double x)
{
    return 1.0f / (1.0f + exp(-x));
}

inline double Relu(double x)
{
    return (x > 0) ? x : 0;
}

Network *CreateAndInit(int iInputSize, int iOutputSize, NetDeclareTable *pxTable)
{
    Network *pxNet = malloc(sizeof(Network));

    pxNet->pxLayerArray = malloc(sizeof(Layer) * pxTable->iLayerNumber);
    pxNet->iLayerArraySize = pxTable->iLayerNumber;

    for (int i = 0; i < pxNet->iLayerArraySize; i++)
    {
        Layer *pxLayer = pxNet->pxLayerArray + i;
        
        /*  */
        pxLayer->cLayerType = (char)*pxTable->pxLayerTable->sLayerType;

    }
    

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



void Forward(Network *pxNet, double *pdInput)
{
    /* 每一层 */
    for (int iLayerIndex = 0; iLayerIndex < pxNet->iLayerArraySize; iLayerIndex++)
    {
        Layer *pxLayer = pxNet->pxLayerArray + iLayerIndex;
        /* 每一个节点 */
        for (int iNodeIndex = 0; iNodeIndex < pxLayer->iNodeArraySize; iNodeIndex++)
        {
            Node *pxNode = pxLayer->pxNodeArray + iNodeIndex;


        }
        
    }
    
}


void Backward()
{

}


#define LAYER_NUM 4

LayerDeclareTable axLayerTable[LAYER_NUM] = {
    {.sLayerType = "Flatten", .iNodeNumber = 128, .sActivateFunType = "Relu"},
    {.sLayerType = "Flatten", .iNodeNumber = 128, .sActivateFunType = "Relu"},
    {.sLayerType = "Flatten", .iNodeNumber = 128, .sActivateFunType = "Relu"},
    {.sLayerType = "Flatten", .iNodeNumber = 128, .sActivateFunType = "Relu"},
};

int main()
{
    /* code */
    printf("ok\n");

    NetDeclareTable xNetTable = {
        .iLayerNumber = LAYER_NUM,
        .pxLayerTable = axLayerTable    
    };

    return 0;
}

#ifndef __BMP_H
#define __BMP_H

#include <stdint.h>

/// @brief 根据数据生成一个灰度位图文件
/// @param strFileName 文件名
/// @param pcGrayImageData 灰度数据
/// @param iWidth   图像宽度
/// @param iHeight  图像高度
/// @return 0:完成无错误 -1:运行失败
int GenerateGrayBitMapFile(const char *strFileName, uint8_t *pcGrayImageData, int iWidth, int iHeight);

#endif /* __BMP_H */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>
#include <stdint.h>
#include "rlwe_sife.h"

#define TERMS 3
#define SEC_LEVEL 1
#define UNKNOWN		32

void polynomial(uint32_t terms[TERMS][2], double number) {
	int integ = (int)number;
	double decim = number - integ;
    
    if (number >= 0) {
		terms[0][0] = integ;
		terms[0][1] = 0;
	} else {
		terms[0][0] = 0;
		terms[0][1] = integ * -1;
	}

    for (int t = 1; t < TERMS; t++) {
        decim = (decim - (int)decim) * UNKNOWN;

        if (number >= 0) {
            terms[t][0] = (int)fabs(decim);
            terms[t][1] = 0;
        } else {
            terms[t][0] = 0;
            terms[t][1] = (int)fabs(decim);
        }
    }
}


double reverse_polynomial(const uint32_t terms[TERMS][2]) {
    double result = 0.0;
    double base = (double)UNKNOWN;

	result += ((double)terms[0][0] - (double)terms[0][1]);

    for (int i = 1; i < TERMS; i++) {
        int value = (double)terms[i][0] - (double)terms[i][1];
        result += value / pow(base, i);
    }

    return result;
}


float* slice(float* parent, int parentSizeX, int* startPoint, int* endPoint) {
	int width = endPoint[0] - startPoint[0] + 1;
    int height = endPoint[1] - startPoint[1] + 1;
    float* result = (float*)calloc(width * height, sizeof(float));

    int point = 0;
    for (int i = startPoint[1]; i <= endPoint[1]; i++) {
        for (int j = startPoint[0]; j <= endPoint[0]; j++) {
            result[point++] = parent[i * parentSizeX + j];
        }
    }

    return result;
}

float arr(float* arr, int dim, int* size, int* pos) {
	int realPos = 0;
	int isValid = 1;
	for (int d = 0; d < dim; d++) {
		if (pos[d] >= size[d]) {
			isValid = 0;
		}
		int temp = 1;
		for (int t = d + 1; t < dim; t++) {
			temp *= size[t];
		}
		temp *= pos[d];
		realPos += temp;
	}
	if (isValid == 0) {
		printf("[ERROR] ?? ??: pos? size?? ??? ???.");
	}
	printf("%f\n", arr[realPos]);
	return arr[realPos];
}

float matrix_product(float* array1, float* array2, int size1, int size2)
{
	float result = 0.0f;
	int seq=0;
		
	for(int i=0 ; i < size1 ; i++)
	{
		for(int j=0 ; j < size2 ; j++)
		{
			seq = i * size2 + j;
			result += array1[seq] * array2[seq];
		}
	}		
	return result;
}

float* convolution(float* image, int* imageSize, float* filter, int* filterSize, int stride) {
	int inputBatch = imageSize[0];
	int inputChannel = imageSize[1];
	int inputWidth = imageSize[2];
	int inputHeight = imageSize[3];

	int filterCount = filterSize[0];
	int filterChannel = filterSize[1];
	int filterWidth = filterSize[2];
	int filterHeight = filterSize[3];

	int outputWidth = floor((inputWidth - filterWidth) / stride) + 1;
	int outputHeight = floor((inputHeight - filterHeight) / stride) + 1;

	float* slicedImage = (float*)calloc(filterWidth * filterHeight, sizeof(float));
	float* slicedFilter = (float*)calloc(filterWidth * filterHeight, sizeof(float));
	float* output = (float*)calloc(inputBatch * filterCount * outputWidth * outputHeight, sizeof(float));

	int pointer = 0;
	int input_W_dot_H = inputWidth * inputHeight;
	int filter_W_dot_H = filterWidth * filterHeight;
	int inputBase = inputChannel * input_W_dot_H;
	int filterBase = filterChannel * filter_W_dot_H;

	for (int b = 0; b < inputBatch; b++) {
		for (int ft = 0; ft < filterCount; ft++) {
			for (int j = 0; j < outputWidth; j++) {
				for (int i = 0; i < outputHeight; i++) {
					float outputPoint = 0;
					int j_dot_stride = j * stride;
					int i_dot_stride = i * stride;
					for (int ch = 0; ch < inputChannel; ch++) {

						for (int fh = 0; fh < filterHeight; fh++) {
							for (int fw = 0; fw < filterWidth; fw++) {
								slicedImage[fh * filterWidth + fw] = image[b * inputBase + ch * input_W_dot_H + inputWidth * (fh + j_dot_stride) + fw + i_dot_stride];
								slicedFilter[fh * filterWidth + fw] = filter[ft * filterBase + ch * filter_W_dot_H + fh * filterWidth + fw];
							}
						}

						outputPoint += matrix_product(slicedImage, slicedFilter, filterWidth, filterHeight);
					}
					output[pointer++] = outputPoint;
				}
			}
		}
	}
	
	return output;
}

float* convolution1x1(float* image, int* imageSize, float* filter, int* filterSize, int stride) {
	uint32_t sk_y[TERMS][2][SIFE_NMODULI][SIFE_N];
	mpz_t dy[SIFE_N];
	double term[TERMS*TERMS][4] = {0};

	uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N];
	uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N];
	uint32_t m[SIFE_L];
	uint32_t y[SIFE_L];
	uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N];

	int inputBatch = imageSize[0];
	int inputChannel = imageSize[1];
	int inputWidth = imageSize[2];
	int inputHeight = imageSize[3];

	int filterCount = filterSize[0];
	int filterLength = filterSize[1];

	int* slicedInput = (int*)calloc(1 * filterLength * TERMS * 2, sizeof(int));
	int* slicedFilter = (int*)calloc(1 * filterLength * TERMS * 2, sizeof(int));

	int outp[filterLength][TERMS];

	int* slicedInputChannel = (int*)calloc(filterLength, sizeof(int));
	int* slicedFilterChannel = (int*)calloc(filterLength, sizeof(int));

	float* output = (float*)calloc(inputBatch * filterCount * inputWidth * inputHeight, sizeof(float));

	int outputWidth = floor((inputWidth - 1) / stride) + 1;
	int outputHeight = floor((inputHeight - 1) / stride) + 1;

	int pointer = 0;
	int input_W_dot_H = inputWidth * inputHeight;
	int inputBase = inputChannel * input_W_dot_H;

	uint32_t polyInput[TERMS][2] = {0};
	uint32_t polyFilter[TERMS][2] = {0};

	// rlwe_sife_setup(mpk, msk);

	for (int b = 0; b < inputBatch; b++) {
		for (int fc = 0; fc < filterCount; fc++) {
			for (int j = 0; j < outputHeight; j++) {
				for (int i = 0; i < outputWidth; i++) {

					// 결과 픽셀 하나를 생성하는 프로세스 시작...

					int sliceInputPointer = 0;
					int sliceFilterPointer = 0;

					// input, filter를 다항화하는 프로세스.

					printf("output pix: (%d %d), filterCount: %d\n", i, j, fc);

					for (int ich = 0; ich < inputChannel; ich++) {
						polynomial(polyInput, image[b * inputBase + ich * input_W_dot_H + inputWidth * j * stride + i * stride]); // 입력 픽셀 실수 하나를 32진법으로 다항화
						polynomial(polyFilter, filter[fc * filterLength + ich]); // 필터 픽셀 실수 하나를 32진법으로 다항화
						for (int poly = 0; poly < TERMS; poly++) {
							slicedInput[poly * inputChannel + ich] = polyInput[poly][0];
							slicedInput[(poly + TERMS) * inputChannel + ich] = polyInput[poly][1];
							slicedFilter[poly * inputChannel + ich] = polyFilter[poly][0];
							slicedFilter[(poly + TERMS) * inputChannel + ich] = polyFilter[poly][1];
						}
					}

					double outPix = 0;

					for (int ft = 0; ft < TERMS * 2; ft++) {
						for (int it = 0; it < TERMS * 2; it++) {
							// slicedInput, slicedFilter 중에서 당장 내적할 부분을 고르는 프로세스.
							for (int ich = 0; ich < inputChannel; ich++) {
								slicedInputChannel[ich] = slicedInput[it * inputChannel + ich];
								slicedFilterChannel[ich] = slicedFilter[ft * inputChannel + ich];
							}
							double prod = 0;
							int sign = 1;
							if ((ft < TERMS && it >= TERMS) || (ft >= TERMS && it < TERMS)) {
								sign = -1;
							}
							// rlwe_sife_encrypt(slicedInputChannel, mpk, c);
							// rlwe_sife_keygen(slicedFilterChannel, msk, sk_y);
							// rlwe_sife_decrypt_gmp(c, slicedFilterChannel, sk_y, dy);
							// 내적 계산해야 함. (slicedInputChannel * slicedFilterChannel)
							// ch별로 내적을 해서 합산을 하게 됨.
							for (int ich = 0; ich < inputChannel; ich++) {
								prod += slicedInputChannel[ich] * slicedFilterChannel[ich];
							}
							outPix += prod / (pow(UNKNOWN, (it % TERMS) + (ft % TERMS))) * sign;
						}
					}
					output[pointer++] = outPix;
				}
			}
		}
	}
	
	return output;
}

int main() {
	float image[1][2][3][3] = {
		{
			{
				{-1.3, -2.5, -3.2},
				{4, 5, 6},
				{7, 8, 9}
			},
			{
				{-3.22, 6, 9.3},
				{12, 15, 18},
				{21, 24, 27}
			}
		}
	};
	float filter[3][2][1][1] = {
		{
			{
				{1.2},
			},
			{
				{4},
			}
		},
		{
			{
				{2}
			},
			{
				{8}
			}
		},
		{
			{
				{3}
			},
			{
				{6}
			}
		}
	};

	// {1, 3} * {1, 4} + {2, 6} * {1, 4} + ...
	int imageSize[4] = {1, 2, 3, 3};
	int filterSize[4] = {3, 2, 1, 1};

	float* imagePtr = &image[0][0][0][0];
	float* filterPtr = &filter[0][0][0][0];

	float* convResult = convolution1x1(imagePtr, imageSize, filterPtr, filterSize, 1);

	for (int i = 0; i < imageSize[0] * imageSize[2] * imageSize[3] * filterSize[0]; i++) {
		printf("%f ", convResult[i]);
	}
}
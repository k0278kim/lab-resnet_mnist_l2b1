#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GETLENGTH(array) (int)(sizeof(array)/sizeof(*(array)))
#define FOREACH(i,count) for (int i = 0; i < (int)count; ++i)

#define TERMS 3
#define SEC_LEVEL 1
#define UNKNOWN		32

#define LENGTH_KERNEL	5

#define LENGTH_FEATURE0	32
#define LENGTH_FEATURE1	(LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE2	(LENGTH_FEATURE1 >> 1)
#define LENGTH_FEATURE3	(LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
#define	LENGTH_FEATURE4	(LENGTH_FEATURE3 >> 1)
#define LENGTH_FEATURE5	(LENGTH_FEATURE4 - LENGTH_KERNEL + 1)

#define INPUT			1
#define LAYER1			6
#define LAYER2			6
#define LAYER3			16
#define LAYER4			16
#define LAYER5			120
#define OUTPUT          10

#if SEC_LEVEL==1
	static const char SIFE_Q_str[] = "76687145727357674227351553";
	#define SIFE_LOGQ_BYTE 88
	#define SIFE_NMODULI 3
	static const uint64_t SIFE_CRT_CONSTS[SIFE_NMODULI]={0, 206923011, 2935204199007202395};	//*
	static const uint32_t SIFE_MOD_Q_I[SIFE_NMODULI] = {16760833, 2147352577, 2130706433};//*
	#define SIFE_B_x 32		// 32 or 64
	#define SIFE_B_y 32		// 32 or 64
	#define SIFE_L 25		// 25/49 or 9
	#define SIFE_N 4096
	#define SIFE_SIGMA 1
	#define SIFE_P (SIFE_B_x*SIFE_B_y*SIFE_L + 1)
	static const char SIFE_P_str[] = "50241";
	static const char SIFE_SCALE_M_str[]="1526385735302993058007";// floor(q/p)
	static const uint64_t SIFE_SCALE_M_MOD_Q_I[SIFE_NMODULI]={13798054, 441557681, 1912932552};	//*
#endif

int* polynomial(int* terms, double number) {
    double temp = number * UNKNOWN;
    
    if (temp >= 0) {
        terms[0] = round(temp);
        terms[TERMS + 0] = 0;
    } else {
        terms[0] = 0;
        terms[TERMS + 0] = round(fabs(temp));
    }

    for (int t = 1; t < TERMS; t++) {
        temp = (temp - round(temp)) * UNKNOWN;

        if (temp >= 0) {
            terms[t] = round(temp);
            terms[TERMS + t] = 0;
        } else {
            terms[t] = 0;
            terms[TERMS + t] = round(fabs(temp));
        }
    }

    return terms;
}


double reverse_polynomial(const int* terms) {
    double result = 0.0;
    double base = (double)UNKNOWN;

    for (int i = 0; i < TERMS; i++) {
        int value = terms[i] - terms[TERMS + i];
        result += value / pow(base, i + 1);
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
		printf("[ERROR] 범위 초과: pos가 size보다 작아야 합니다.");
	}
	printf("%f\n", arr[realPos]);
	return arr[realPos];
}

float matrix_product(float* array1, float* array2, int size1, int size2)
{
	float result = 0.0f;
	int seq=0;
	
	//result = malloc(sizeof(double) * size);
	
	for(int i=0 ; i < size1 ; i++)
	{
		for(int j=0 ; j < size2 ; j++)
		{
			seq = i * size2 + j;
			result += array1[seq] * array2[seq];
			
			//printf("%f * %f = %f\n", array1[seq], array2[seq], result);
		}
	}
	
	//printf("\n");
		
	return result;
}

// image 크기, 출력 크기, 채널, 필터 크기, stride
// ✅ BATCH 추가본.
float* convolution(float* image, int* imageSize, float* filter, int* filterSize, int stride) {
	// input size: {batch, input channel, width, height}
	// filter size: {filter count, channel, width, height}
	// output size: {width, height, output channel(=filter count)}

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

	// 결과 픽셀 포인트를 기준으로 현재 연산하고 있는 픽셀 인덱스를 알려주는 변수.
	int pointer = 0;
	int input_W_dot_H = inputWidth * inputHeight;
	int filter_W_dot_H = filterWidth * filterHeight;
	int inputBase = inputChannel * input_W_dot_H;
	int filterBase = filterChannel * filter_W_dot_H;

	for (int b = 0; b < inputBatch; b++) {
		for (int ft = 0; ft < filterCount; ft++) {
			for (int j = 0; j < outputWidth; j++) {
				for (int i = 0; i < outputHeight; i++) {
					// outputPoint: 결과 이미지 픽셀 한 개를 임시로 저장한다.
					float outputPoint = 0;
					int j_dot_stride = j * stride;
					int i_dot_stride = i * stride;
					for (int ch = 0; ch < inputChannel; ch++) {
						// 슬라이스를 모두 제거하고 인덱스로 접근하게 만들었더니 성능이
						// 1.81s/it => 0.0377s/it로 향상되었다!!!
						// [Image]
						// 1. b * inputWidth * inputHeight * inputChannel: image 데이터에서 현재 배치가 시작되는 부분의 인덱스.
						// 2. ch * inputWidth * inputHeight: 현재 채널의 시작점 인덱스. (1+2: 현재 배치에서 현재 채널의 시작점 인덱스.)
						// 3. inputWidth * (fh + j * stride): 
						// 4. fw + i * stride: (3+4: 현재 채널에서 자를 부분을 하나씩 집어가는 인덱스.)
						// [Filter]
						// 1. ft * filterWidth * filterHeight * filterChannel: filter 데이터에서 현재 filter가 시작되는 부분의 인덱스.
						// 2. ch * filterWidth * filterHeight: 현재 채널의 시작점 인덱스. (1+2: 현재 filter에서 현재 채널의 시작점 인덱스.)
						// 3. fh * filterWidth + fw: 2.(현재 채널의 시작점 인덱스)부터 하나씩 집어가는 인덱스.(현재 채널 시작점 ~ 현재 채널 끝점)

						for (int fh = 0; fh < filterHeight; fh++) {
							for (int fw = 0; fw < filterWidth; fw++) {
								slicedImage[fh * filterWidth + fw] = image[b * inputBase + ch * input_W_dot_H + inputWidth * (fh + j_dot_stride) + fw + i_dot_stride];
								slicedFilter[fh * filterWidth + fw] = filter[ft * filterBase + ch * filter_W_dot_H + fh * filterWidth + fw];
							}
						}

						outputPoint += matrix_product(slicedImage, slicedFilter, filterWidth, filterHeight);
					}
					// 한 위치에서 모든 채널을 내적 완료했으면, 누적된 결과값을 결과 픽셀 하나에 저장한다.
					// 한 개의 필터를 이미지 한 부분의 모든 채널과 내적하였으며, 한 개의 필터와 한 개의 이미지를 내적한 결과값의 한 픽셀포인트 연산을 완료하였다.
					output[pointer++] = outputPoint;
				}
			}
		}
	}
	
	return output;
}

float* convolution1x1(float* image, int* imageSize, float* filter, int* filterSize, int stride) {

	int inputBatch = imageSize[0];
	int inputChannel = imageSize[1];
	int inputWidth = imageSize[2];
	int inputHeight = imageSize[3];

	int filterCount = filterSize[0]; // filter 개수
	int filterLength = filterSize[1]; // filter 채널

	float* slicedImage = (float*)calloc(1 * filterLength, sizeof(float));
	float* slicedFilter = (float*)calloc(1 * filterLength, sizeof(float));
	float* output = (float*)calloc(inputBatch * filterCount * inputWidth * inputHeight, sizeof(float));

	int outputWidth = floor((inputWidth - 1) / stride) + 1;
	int outputHeight = floor((inputHeight - 1) / stride) + 1;

	int pointer = 0;
	int filterPointer = 0;
	int input_W_dot_H = inputWidth * inputHeight;
	int inputBase = inputChannel * input_W_dot_H;

	int* polyInput = (int*)calloc(TERMS * 2, sizeof(int));
	int* polyFilter = (int*)calloc(TERMS * 2, sizeof(int));

	for (int b = 0; b < inputBatch; b++) {
		for (int fc = 0; fc < filterCount; fc++) {
			for (int j = 0; j < outputHeight; j++) {
				for (int i = 0; i < outputWidth; i++) {
					int slicePointer = 0;
					// slicedImage: image를 같은 위치 포인트에서 채널을 뭉친 것. shape: (1, inputChannel(=filterLength))
					for (int ich = 0; ich < inputChannel; ich++) {
						polynomial(polyInput, image[b * inputBase + ich * input_W_dot_H + inputWidth * j * stride + i * stride]);
						polynomial(polyFilter, filter[fc * filterLength + ich]);
						slicedImage[slicePointer++] = reverse_polynomial(polyInput);
						slicedFilter[ich] = reverse_polynomial(polyFilter);
					}
					// slicedFilter: 1x1 filter의 채널을 뭉친 것. shape: (1, filterLength)
					// 함수 암호 메서드 여기서 호출하기. (slicedImage, slicedFilter 적용 완료)
					// outputPoint는 한 픽셀 위치에서의 여러 채널을 뭉쳐서 내적한 결과.
					output[pointer++] = matrix_product(slicedImage, slicedFilter, filterLength, 1);
				}
			}
		}
	}
	
	return output;
}

// float* convolution1x1(float* image, int* imageSize, float* filter, int* filterSize, int stride) {

// 	int inputBatch = imageSize[0];
// 	int inputChannel = imageSize[1];
// 	int inputWidth = imageSize[2];
// 	int inputHeight = imageSize[3];

// 	int filterCount = filterSize[0]; // filter 개수
// 	int filterLength = filterSize[1]; // filter channel

// 	float* slicedImage = (float*)calloc(1 * filterLength * TERMS, sizeof(float));
// 	float* slicedFilter = (float*)calloc(1 * filterLength * TERMS, sizeof(float));
// 	float* output = (float*)calloc(inputBatch * filterCount * inputWidth * inputHeight, sizeof(float));

// 	int outputWidth = floor((inputWidth - 1) / stride) + 1;
// 	int outputHeight = floor((inputHeight - 1) / stride) + 1;

// 	int pointer = 0;
// 	int filterPointer = 0;
// 	int input_W_dot_H = inputWidth * inputHeight;
// 	int inputBase = inputChannel * input_W_dot_H;

// 	int* polyInput = (int*)calloc(TERMS, sizeof(int));
// 	int* polyFilter = (int*)calloc(TERMS, sizeof(int));

// 	for (int b = 0; b < inputBatch; b++) {
// 		for (int fc = 0; fc < filterCount; fc++) {
// 			for (int j = 0; j < outputHeight; j++) {
// 				for (int i = 0; i < outputWidth; i++) {
// 					int slicePointer = 0;
// 					// printf("%d /// ", pointer);
// 					// slicedImage: image를 같은 위치 포인트에서 채널을 뭉쳐서 다항화한 것. shape: (TERMS, inputChannel(=filterLength))
// 					// slicedFilter: 한 filter의 채널을 뭉쳐서 다항화한 것. shape: (TERMS, filterLength)
// 					for (int ich = 0; ich < inputChannel; ich++) {
// 						polynomial(polyInput, image[b * inputBase + ich * input_W_dot_H + inputWidth * j * stride + i * stride]);
// 						polynomial(polyFilter, filter[fc * filterLength + ich]);
// 						for (int p = 0; p < TERMS; p++) {
// 							slicedImage[slicePointer++] = (float)polyInput[p];
// 							slicedFilter[ich * filterLength + p] = (float)polyFilter[p];
// 						}
// 						// printf("[rev_poly] slicedImage: %f and slicedFilter: %f\n", reverse_polynomial(polyInput), reverse_polynomial(polyFilter));
// 					}
// 					// 함수 암호 메서드 여기서 호출하기. (slicedImage, slicedFilter 적용 완료)
// 					// outputPoint는 한 픽셀 위치에서의 여러 채널을 뭉쳐서 내적한 결과.
// 					output[pointer++] = matrix_product(slicedImage, slicedFilter, filterLength, 1);
// 				}
// 			}
// 		}
// 	}

// 	free(slicedImage);
// 	free(slicedFilter);
// 	free(polyInput);
// 	free(polyFilter);
	
// 	return output;
// }
// // 1.3125 - 0.40625 + 0.1875

int main() {
	float image[1][2][3][3] = {
		{
			{
				{-1.3, 2.5, 3.2},
				{4, 5, 6},
				{7, 8, 9}
			},
			{
				{-3.22, 6, 9},
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
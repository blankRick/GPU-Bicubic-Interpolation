#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#define checkCudaErrors(x) if(x != cudaSuccess) {fprintf(stderr, "cuda Failed\n"); exit(0);}

texture<uchar2, 2, cudaReadModeNormalizedFloat> tex;
cudaArray *d_imageArray = 0;

inline __device__
float w0(float a)
{
	    return (1.0f/6.0f)*((1-a)*(1-a)*(1-a)); // optimized
	//return (1.0f / 6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   
}

inline  __device__
float w1(float a)
{
	//return (2.0f/3.0f) - (1.0f/2.0f) * a*a * (2-a);
	return (1.0f / 6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

inline __device__
float w2(float a)
{
	//    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
	return (1.0f / 6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

inline __device__
float w3(float a)
{
	return (1.0f / 6.0f)*(a*a*a);
}

// g0 and g1 are the two amplitude functions
inline __device__ float g0(float a)
{
	return w0(a) + w1(a);
}

inline __device__ float g1(float a)
{
	return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
inline __device__ float h0(float a)
{
	// note +0.5 offset to compensate for CUDA linear filtering convention
	return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}

inline __device__ float h1(float a)
{
	return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

// fast bicubic texture lookup using 4 bilinear lookups
// assumes texture is set to non-normalized coordinates, point sampling
template<class T, class R>  // texture data type, return type
__device__
R tex2DFastBicubic(const texture<T, 2, cudaReadModeNormalizedFloat> texref, float x, float y)
{
	x -= 0.5f;
	y -= 0.5f;
	float px = floor(x);
	float py = floor(y);
	float fx = x - px;
	float fy = y - py;

	// note: we could store these functions in a lookup table texture, but maths is cheap
	float g0x = g0(fx);
	float g1x = g1(fx);
	float h0x = h0(fx);
	float h1x = h1(fx);
	float h0y = h0(fy);
	float h1y = h1(fy);

	float2 tex00 = tex2D(texref, px + h0x, py + h0y);
	float2 tex10 = tex2D(texref, px + h1x, py + h0y);
	float2 tex01 = tex2D(texref, px + h0x, py + h1y);
	float2 tex11 = tex2D(texref, px + h1x, py + h1y);

	float rx =	g0(fy) * (g0x * tex00.x +
						  g1x * tex10.x) +
				g1(fy) * (g0x * tex01.x +
						  g1x * tex11.x);
	float ry =	g0(fy) * (g0x * tex00.y +
						  g1x * tex10.y) +
				g1(fy) * (g0x * tex01.y +
						  g1x * tex11.y);
	return make_float2(rx,ry);
}

// render image using fast bicubic texture lookup
__global__ void
d_renderFastBicubic(unsigned char *d_output, unsigned int width, unsigned int height, float tx, float ty, float scale, float cx, float cy)
{
	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	unsigned int i = (y * width) + x;

	float u = (x - cx)*scale + cx + tx;
	float v = (y - cy)*scale + cy + ty;

	if ((x < width) && (y < height))
	{
		// write output color
		float2 c = tex2DFastBicubic<uchar2, float2>(tex, u, v);
		d_output[i] = c.x * 0xff;
		d_output[i + width * height] = c.y * 0xff;
	}
}

void initTexture(int imageWidth, int imageHeight, unsigned char *h_data)
{
	// allocate array and copy image data
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned);
	checkCudaErrors(cudaMallocArray(&d_imageArray, &channelDesc, imageWidth, imageHeight));
	unsigned int size = imageWidth * imageHeight * 2 * sizeof(unsigned char);
	checkCudaErrors(cudaMemcpyToArray(d_imageArray, 0, 0, h_data, size, cudaMemcpyHostToDevice));
	//free(h_data);

	// set texture parameters
	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false;    // access with integer texture coordinates


	// Bind the array to the texture
	checkCudaErrors(cudaBindTextureToArray(tex, d_imageArray));
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(float* rearrangedData, unsigned char* output, int pad)
{
	int rearrangedId = gridDim.y * (threadIdx.x * gridDim.x);

	output[threadIdx.x * (gridDim.y * gridDim.x) + blockIdx.x * gridDim.y + blockDim.y] = (unsigned char)(fmaxf(0, fminf(255, rearrangedData[blockDim.x * ((blockIdx.x + pad) * (gridDim.y + 2*pad) + blockIdx.y) + threadIdx.x])));
}

int main()
{
	int inputWidth = 3840;
	int inputHeight = 2160;

    // Add vectors in parallel.
	cudaError_t cudaStatus = cudaSuccess;
    //cudaStatus = addWithCuda(c, a, b, arraySize);
	unsigned char * buffer1 = (unsigned char*)calloc(inputWidth * inputHeight / 2, sizeof(unsigned char));
	unsigned char * buffer2 = (unsigned char*)calloc(inputWidth * inputHeight / 2, sizeof(unsigned char));
	unsigned char * d_output1;
	unsigned char * h_output1 = (unsigned char*)calloc(inputWidth * inputHeight * 2, sizeof(unsigned char));
	cudaMallocManaged(&d_output1, inputWidth * inputHeight * 2 * sizeof(unsigned char));
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}
	FILE * fp = fopen("D:\\Downloads\\shelter_3840x2160_422p.YUV", "rb");
	if (fp == NULL) { fprintf(stderr, "File not found\n"); exit(0); }
	fread(buffer1, inputWidth * inputHeight / 4, sizeof(unsigned char), fp);
	fread(buffer1, inputWidth * inputHeight / 4, sizeof(unsigned char), fp);
	fread(buffer1, inputWidth * inputHeight / 4, sizeof(unsigned char), fp);
	fread(buffer1, inputWidth * inputHeight / 4, sizeof(unsigned char), fp);
	fread(buffer1, inputWidth * inputHeight / 2, sizeof(unsigned char), fp);
	for (int i = 0; i < inputWidth * inputHeight / 4; i++)
	{
		buffer2[i * 2 + 0] = buffer1[i];
		buffer2[i * 2 + 1] = buffer1[i + inputWidth * inputHeight/4];
	}
	fclose(fp);

	time_t start, end;
	dim3 blockSize(480, 1);
	dim3 gridSize(inputWidth / blockSize.x, inputHeight / blockSize.y);

	double elapsedTime=0;
	for (int i = 0; i < 50; i++)
	{
		initTexture(inputWidth / 2, inputHeight / 2, buffer2);

		start = clock();
		//cudaEventRecord(startEvent);
		d_renderFastBicubic << <gridSize, blockSize>> >(d_output1, inputWidth, inputHeight, -inputWidth / 4.0f, -inputHeight / 4.0f, 1 / 2.0f, inputWidth / 2.0f, inputHeight / 2.0f);

		cudaStreamSynchronize(0);
		end = clock();
		//cudaMemcpy(h_output1, d_output1, inputWidth * inputHeight * 2 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		if(i>2) elapsedTime += (double)(end - start) / CLOCKS_PER_SEC;
		//cudaEventRecord(endEvent);
	}
	//cudaEventElapsedTime(&elapsedTime, startEvent, endEvent);
	printf("Elapsed time = %lf s\n", elapsedTime );
	//printf("Elapsed time = %lf s\n", (double)elapsedTime);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}

	FILE * wfp = fopen("D:\\Downloads\\shelter_3840x2160_400p.YUV", "wb");
	if (wfp == NULL) { fprintf(stderr, "File not found\n"); exit(0); }
	for (int i = 0; i < 10; i++)
	{
		printf("%d ", (int)h_output1[i]);
	}
	printf("\n");
	fwrite(d_output1, sizeof(unsigned char), inputWidth * inputHeight * 2, wfp);
	printf("Written\n");
	fclose(wfp);
	//initTexture(inputWidth / 2, inputHeight / 2, buffer2);


    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	printf("Reached here\n");
    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
       // c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
/*
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/
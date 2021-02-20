
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdio.h>
using namespace std;


typedef struct {
	unsigned long long int size;
	float* elementyFloat;
	double* elementyDouble;
}Tablica;


Tablica generuj(char decyzja);
float LeibnizCPUF(Tablica tab);
float LeibnizGPUF(Tablica tab, double &czasGPU);
float LeibnizGPUF_bez_shared(Tablica tab, double &czasGPU);

double LeibnizCPUD(Tablica tab);
double LeibnizGPUD(Tablica tab, double &czasGPU);
double LeibnizGPUD_bez_shared(Tablica tab, double &czasGPU);

__global__ void kernelDouble(double* in_temp, double* out_temp, unsigned long long int n);
__global__ void kernelFloat(float* in_temp, float* out_temp, unsigned long long int n);

__global__ void kernelBezSharedDouble(double* in_temp, double* out_temp, unsigned long long int n);
__global__ void kernelBezSharedFloat(float* in_temp, float* out_temp, unsigned long long int n);



int main()
{

	// kod na GPU wzorowany na poradniku nvidii dot. dodawania równoleg³ego : http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
	char decyzja;
	cout << "Float czy Double? F/D" << endl;
	cin >> decyzja;
	Tablica vals = generuj(decyzja);
	double PI = 3.141592653589793238462643383279502884197169399375105820974944592307;



	cout << "---------------------------" << endl;
	cout << setprecision(30) << "WZOR LICZBY PI: " << PI << endl;
	cout << "---------------------------" << endl;



	if (decyzja == 'f' || decyzja == 'F')
	{
		double czasShared;
		double czasBezShared;
		auto CPUstart = chrono::steady_clock::now();
		float zmienna = LeibnizCPUF(vals);
		auto CPUend = chrono::steady_clock::now();
		chrono::duration<double> elapsedCPU = CPUend - CPUstart;


		cout << "---------------------------" << endl;
		cout << "Czas CPU: " << elapsedCPU.count() << endl;
		cout << "Cpu dla float : " << zmienna << endl;
		cout << "---------------------------" << endl;

		float zmienna2 = LeibnizGPUF_bez_shared(vals, czasBezShared);
		cout << "GPU bez pamieci shared: " << zmienna2 << endl;

		cout << "---------------------------" << endl;
		float gpuf = LeibnizGPUF(vals, czasShared);
		cout << "GPU dla float : " << gpuf << endl;
		cout << "---------------------------" << endl;
		
		cout << "Przyspieszenie Bez Shared/CPU: " << elapsedCPU.count() / czasBezShared << endl;
		cout << "Przyspieszenie Shared/CPU: " << elapsedCPU.count() / czasShared << endl;
		cout << "Przyspieszenie Shared/Bez Shared: " << czasBezShared / czasShared << endl;
		cout << "---------------------------" << endl;
		cout << "Blad CPU: " << PI - zmienna << endl;
		cout << "Blad GPU: " << PI - zmienna2 << endl;
		cout << "Blad GPU z uzyciem shared: " << PI - gpuf << endl;
		delete vals.elementyFloat;
		
	}
	else
	{
		double czasShared=0;
		double czasBezShared=0;
		auto CPUstart = chrono::steady_clock::now();
		double zmienna = LeibnizCPUD(vals);
		auto CPUend = chrono::steady_clock::now();
		chrono::duration<double> elapsedCPU = CPUend - CPUstart;

		cout << "---------------------------" << endl;
		cout << "Czas CPU: " << elapsedCPU.count() << endl;
		cout <<"Cpu dla double : " << zmienna << endl;
		

		cout << "---------------------------" << endl;
		double zmienna2 = LeibnizGPUD_bez_shared(vals, czasBezShared);
		cout << "GPU bez pamieci shared: " << zmienna2 << endl;


		cout << "---------------------------" << endl;
		double gpud = LeibnizGPUD(vals, czasShared);
		cout << "GPU dla double : " << gpud << endl;
		
		cout << "---------------------------" << endl;
		cout << "Przyspieszenie Bez Shared/CPU: " <<  elapsedCPU.count()/czasBezShared << endl;
		cout << "Przyspieszenie Shared/CPU: " << elapsedCPU.count()/czasShared << endl;
		cout << "Przyspieszenie Shared/Bez Shared: " << czasBezShared / czasShared << endl;
		cout << "---------------------------" << endl;
		cout << "Blad CPU: " << PI - zmienna << endl;
		cout << "Blad GPU: " << PI - zmienna2 << endl;
		cout << "Blad GPU z uzyciem shared: " << PI - gpud << endl;
		delete vals.elementyDouble;
	}
}

Tablica generuj(char decyzja)
{
	Tablica tablica = Tablica();

	float free_m;
	size_t free_t, total_t;
	cudaMemGetInfo(&free_t, &total_t);
	free_m = (unsigned long long int)free_t / sizeof(double);
	tablica.size = free_m;
	

	unsigned long long int NR_BLOCKS = static_cast<unsigned long long int>(tablica.size / 512) + 1;
	cout << "Tablica.size przed redukcja: " << tablica.size << endl;
	tablica.size = tablica.size - NR_BLOCKS * sizeof(double)-7*10000000;
	cout << "Tablica.size po redukcji: " << tablica.size << endl;
	//tablica.size = 10000000;

	if (decyzja == 'F' || decyzja == 'f')
	{
		tablica.size = tablica.size * 2 - 10000;
		tablica.elementyFloat = new float[tablica.size];


		for (unsigned long long int i = 0; i < tablica.size; i++)
		{
			if (i % 2 == 0)
			{
				tablica.elementyFloat[i] = 4 * float(1 / (2 * float(i) + 1));
			}
			else
			{
				tablica.elementyFloat[i] = 4 * float(-1 / (2 * float(i) + 1));
			}
		}

		return tablica;
	}
	else if (decyzja == 'D' || decyzja == 'd')
	{
		tablica.elementyDouble = new double[tablica.size];
		for (unsigned long long int i = 0; i < tablica.size; i++)
		{
			if (i % 2 == 0)
			{
				tablica.elementyDouble[i] = double(4 * 1 / (2 * double(i) + 1));
			}
			else
			{
				tablica.elementyDouble[i] = double(4 * -1 / (2 * double(i) + 1));
			}
		}

		return tablica;
	}
	else return tablica;
}

float LeibnizCPUF(Tablica tab)
{
	float c = 0;
	for (unsigned long long int i = 0; i < tab.size; i++)
	{
		c = c + tab.elementyFloat[i];
	}
	return c;
}
double LeibnizCPUD(Tablica tab)
{
	double c = 0;
	for (unsigned long long int i = 0; i < tab.size; i++)
	{
		c = c + tab.elementyDouble[i];
	}
	return c;
}

double LeibnizGPUD(Tablica tab, double &czasGPU)
{
	unsigned long long int NR_BLOCKS = static_cast<unsigned long long int>(tab.size / 512) + 1;
	//cout << "Numer blokow " << NR_BLOCKS << endl;
	//cout << "Rozmiar " << tab.size << endl;
	
	double* out_temp;
	double* d_out_temp;
	out_temp = new double[NR_BLOCKS];

	double* in_temp;
	double* d_in_temp;
	in_temp = tab.elementyDouble;


	unsigned long long int n = tab.size;


	unsigned long long int size = tab.size * sizeof(double);
	unsigned long long int sizeout = NR_BLOCKS * sizeof(double);

	cudaMalloc((void**)&d_in_temp, size);
	cudaMalloc((void**)&d_out_temp, sizeout );
	
		
	cudaMemcpy(d_in_temp, in_temp, size, cudaMemcpyHostToDevice);
	auto GPUstart = chrono::steady_clock::now();
		
	kernelDouble << <NR_BLOCKS, 512 >> > (d_in_temp, d_out_temp,n);
	n = NR_BLOCKS;
	kernelDouble << <1, 512 >> > (d_out_temp, d_out_temp,n);
	auto GPUend = chrono::steady_clock::now();
	chrono::duration<double> elapsedGPU = GPUend - GPUstart;
	cout << "Czas GPU: " << elapsedGPU.count() << endl;
	czasGPU = elapsedGPU.count();
	cudaMemcpy(out_temp, d_out_temp, sizeout, cudaMemcpyDeviceToHost);
		
	double z = out_temp[0];
	delete(out_temp);
	cudaFree(d_in_temp); cudaFree(d_out_temp);
	return z;
}

float LeibnizGPUF(Tablica tab,  double &czasGPU)
{
	unsigned long long int NR_BLOCKS = static_cast<unsigned long long int>(tab.size / 512) + 1;
	

	float* out_temp;
	float* d_out_temp;
	out_temp = new float[NR_BLOCKS];

	float* in_temp;
	float* d_in_temp;
	in_temp = tab.elementyFloat;


	unsigned long long int n = tab.size;


	unsigned long long int size = tab.size * sizeof(float);
	unsigned long long int sizeout = NR_BLOCKS * sizeof(float);

	cudaMalloc((void**)&d_in_temp, size);
	cudaMalloc((void**)&d_out_temp, sizeout);


	cudaMemcpy(d_in_temp, in_temp, size, cudaMemcpyHostToDevice);
	
	auto GPUstart = chrono::steady_clock::now();

	kernelFloat << <NR_BLOCKS, 512 >> > (d_in_temp, d_out_temp, n);
	n = NR_BLOCKS;
	kernelFloat << <1, 512 >> > (d_out_temp, d_out_temp, n);

	auto GPUend = chrono::steady_clock::now();
	chrono::duration<double> elapsedGPU = GPUend - GPUstart;
	
	cout << "Czas GPU: " << elapsedGPU.count() << endl;
	cudaMemcpy(out_temp, d_out_temp, sizeout, cudaMemcpyDeviceToHost);
	czasGPU = elapsedGPU.count();
	float z = out_temp[0];
	delete(out_temp);
	cudaFree(d_in_temp); cudaFree(d_out_temp);
	return z;
}

double LeibnizGPUD_bez_shared(Tablica tab,  double &czasGPU)
{
	unsigned long long int rozmiar = tab.size;
	unsigned long long int NR_BLOCKS = static_cast<unsigned long long int>(tab.size / 512) + 1;
	
	double* out_temp;
	double* d_out_temp;
	out_temp = new double[(rozmiar/2)+1];

	double* in_temp;
	double* d_in_temp;
	in_temp = tab.elementyDouble;

	unsigned long long int size = tab.size * sizeof(double);
	unsigned long long int sizeout = ((rozmiar/2) +1) * sizeof(double);

	cudaMalloc((void**)&d_in_temp, size);
	cudaMalloc((void**)&d_out_temp, sizeout);

	cudaMemcpy(d_in_temp, in_temp, size, cudaMemcpyHostToDevice);
	auto GPUstart = chrono::steady_clock::now();

	kernelBezSharedDouble << <NR_BLOCKS, 512>>>(d_in_temp, d_out_temp, rozmiar);
	rozmiar = (rozmiar / 2);
	NR_BLOCKS = (rozmiar / 512) + 1;
	while (rozmiar > 1)
	{
		kernelBezSharedDouble << <NR_BLOCKS, 512 >> > (d_out_temp, d_out_temp, rozmiar);
		rozmiar = (rozmiar / 2);
		NR_BLOCKS =(rozmiar / 512) + 1;
		
	}
	auto GPUend = chrono::steady_clock::now();
	chrono::duration<double> elapsedGPU = GPUend - GPUstart;
	cout << "Czas GPU: " << elapsedGPU.count() << endl;
	cudaMemcpy(out_temp, d_out_temp, sizeout, cudaMemcpyDeviceToHost);
	czasGPU = elapsedGPU.count();
	double z = out_temp[0];
	delete(out_temp);
	cudaFree(d_in_temp); cudaFree(d_out_temp);
	return z;
}

float LeibnizGPUF_bez_shared(Tablica tab, double &czasGPU)
{
	unsigned long long int rozmiar = tab.size;
	unsigned long long int NR_BLOCKS = static_cast<unsigned long long int>(tab.size / 512) + 1;

	float* out_temp;
	float* d_out_temp;
	out_temp = new float[(rozmiar / 2) + 1];

	float* in_temp;
	float* d_in_temp;
	in_temp = tab.elementyFloat;

	unsigned long long int size = tab.size * sizeof(float);
	unsigned long long int sizeout = ((rozmiar / 2) + 1) * sizeof(float);

	cudaMalloc((void**)&d_in_temp, size);
	cudaMalloc((void**)&d_out_temp, sizeout);

	cudaMemcpy(d_in_temp, in_temp, size, cudaMemcpyHostToDevice);
	
	auto GPUstart = chrono::steady_clock::now();
	kernelBezSharedFloat << <NR_BLOCKS, 512 >> > (d_in_temp, d_out_temp, rozmiar);
	rozmiar = (rozmiar / 2);
	NR_BLOCKS = (rozmiar / 512) + 1;
	while (rozmiar > 1)
	{
		kernelBezSharedFloat << <NR_BLOCKS, 512 >> > (d_out_temp, d_out_temp, rozmiar);
		rozmiar = (rozmiar / 2);
		NR_BLOCKS = (rozmiar / 512) + 1;

	}
	auto GPUend = chrono::steady_clock::now();
	chrono::duration<double> elapsedGPU = GPUend - GPUstart;
	cout << "Czas GPU: " << elapsedGPU.count() << endl;
	cudaMemcpy(out_temp, d_out_temp, sizeout, cudaMemcpyDeviceToHost);
	czasGPU = elapsedGPU.count();
	float z = out_temp[0];
	delete(out_temp);
	cudaFree(d_in_temp); cudaFree(d_out_temp);
	return z;
}

__global__ void kernelDouble(double *in_temp, double *out_temp, unsigned long long int n)
{	
	int BlockSize = 512;
	__shared__ double sdata[512];
	unsigned int tid = threadIdx.x;
	unsigned int index = threadIdx.x + blockIdx.x * BlockSize*2;
	unsigned int gridSize = BlockSize * 2 * gridDim.x;
	
	sdata[tid] = 0;
	__syncthreads();


	// wstêpne dodawanie do pamieci shared ktore pozwala nam m.in w drugim wywo³aniu kernela sumowaæ wszystkie pozosta³e w¹tki
	while (index < n)
	{
		sdata[tid] += in_temp[index] + in_temp[index + BlockSize];
		index += gridSize;
		
	}
	__syncthreads();
	
	//redukcja
	for (unsigned int s = BlockSize / 2; s > 0; s >>= 1)
	{

		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	//wpisywanko
	if (tid == 0)
	{
		
		out_temp[blockIdx.x] = sdata[0];
	}
}

__global__ void kernelFloat(float* in_temp, float* out_temp, unsigned long long int n)
{
	int BlockSize = 512;
	__shared__ double sdata[512];
	unsigned int tid = threadIdx.x;
	unsigned int index = threadIdx.x + blockIdx.x * BlockSize * 2;
	unsigned int gridSize = BlockSize * 2 * gridDim.x;

	sdata[tid] = 0;
	__syncthreads();


	// wstêpne dodawanie do pamieci shared ktore pozwala nam m.in w drugim wywo³aniu kernela sumowaæ wszystkie pozosta³e w¹tki
	while (index < n)
	{
		sdata[tid] += in_temp[index] + in_temp[index + BlockSize];
		index += gridSize;

	}
	__syncthreads();

	//redukcja
	for (unsigned int s = BlockSize / 2; s > 0; s >>= 1)
	{

		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	//wpisywanko
	if (tid == 0)
	{

		out_temp[blockIdx.x] = sdata[0];
		//printf(" %d", out_temp[blockIdx.x]);
	}
}

__global__ void kernelBezSharedDouble(double* in_temp, double* out_temp, unsigned long long int n)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	double z = 0;
	if (i < n / 2)
	{
		if (n % 2 == 0)
		{
			z = in_temp[i] + in_temp[(n / 2) + i];
			
			out_temp[i] = z;
		}
		else
		{
			if (i == 0)
			{		
				z = in_temp[i] + in_temp[(n / 2) ] + in_temp[n-1];
				out_temp[i] = z;
			}
			else
			{
				z = in_temp[i] + in_temp[(n / 2) + i];
				out_temp[i] = z;
			}
		}
	}
}

__global__ void kernelBezSharedFloat(float* in_temp, float* out_temp, unsigned long long int n)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	float z = 0;
	if (i < n / 2)
	{
		if (n % 2 == 0)
		{
			z = in_temp[i] + in_temp[(n / 2) + i];

			out_temp[i] = z;
		}
		else
		{
			if (i == 0)
			{
				z = in_temp[i] + in_temp[(n / 2)] + in_temp[n - 1];
				out_temp[i] = z;
			}
			else
			{
				z = in_temp[i] + in_temp[(n / 2) + i];
				out_temp[i] = z;
			}
		}
	}
}
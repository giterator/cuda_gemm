// code inspired by: https://github.com/siboehm/SGEMM_CUDA

#include<iostream>
#include<vector>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <cassert>
#include <sys/time.h>
#include <iomanip>
#include <fstream>

using namespace std;

int runs = 100;
vector<int> sizes = {256, 512, 1024, 2048, 4096, 8192};

void naive_mm(float *A, float *B, float *truth, int n){
  for(int k = 0; k < n; k++){
    for(int i = 0; i < n; i++){
      for(int j = 0; j < n; j++){
        truth[i*n+j] += A[i*n+k] * B[k*n+j];
      }
    }
  }
}

void reset_c(float *C, int n){
  for( int i = 0; i < n*n; i++){
    C[i] = 0.0;
  }
}


void check_correctness(float *truth, float *C, int n){
  for(int i = 0; i < n*n; i++){
    float diff = std::fabs(truth[i] - C[i]);
    float relerr = diff / (fabs(truth[i]) + 1e-12);
    if(diff > 1e-3 && relerr > 1e-5){
      cout << "\t Matrices don't match \n";
      cout << "\t\t Row:"<< i/n <<" Col: " << i%n << endl;
      exit(1);
    }
  }

  cout <<"\t Correct!\n";

}

void init_mats(float *A, float *B, float *C, float *truth, int n){
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < n*n; ++i){
      A[i] = static_cast<float>(rand()) / RAND_MAX;
      B[i] = static_cast<float>(rand()) / RAND_MAX;
      C[i] = 0.0;
      truth[i] = 0.0;
    } 
}

void print_mat(const float *M, int n, const char* name, int max_rows = 8, int max_cols = 8) {
    printf("%s (%dx%d):\n", name, n, n);
    int rows = (n < max_rows) ? n : max_rows;
    int cols = (n < max_cols) ? n : max_cols;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.2f ", M[i * n + j]);
        }
        if (cols < n) printf(" ...");
        printf("\n");
    }
    if (rows < n) printf("...\n");
    printf("\n");
}

void save_matrix(const float* mat, int rows, int cols, const char* filename) {
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(2);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file << mat[i * cols + j] << " ";
        }
        file << "\n";
    }
    file.close();
}

constexpr int TH = 32;
constexpr int TW = 4;

constexpr int BW = 256;
constexpr int BH = 128;
constexpr int BK = 16;

constexpr int WW = 64;
constexpr int WH = 64;

 //1 thread computes w threadtiles where each threadtile is THxTW. 
 // w threadtiles are disjoint, spread across the warptile in a 2D manner
constexpr int w = ( (WH*WW) / (TH * TW) ) / 32; // = no. threadtiles in the warp tile / 32 i.e. no. threadtiles computed per thread in a warp
constexpr int w_hor = 1;
constexpr int w_vert =  w/w_hor;

constexpr int threads_per_block = BW/WW * BH/WH * 32; // 1 warp computes 1 warptile


__global__ void kernel(float *A, float* B, float* C, int n){
  
  // register level cache for output tile
 float c_out[w * TH * TW] = {0.0};

 // SM level cache for reuse in thread block & enable coalesced GMem access when populating
 __shared__ float a[BK * BH];
 __shared__ float b[BK * BW];

 // thread level cache for reuse in warp tile & enable warp level broadcast when populating
 float reg_a[w_vert * TH];// no. vertical thread tiles worth of rows
 float reg_b[w_hor * TW]; // no. horizontal thread tiles worth of cols

 int block_col = blockIdx.x;
 int block_row = blockIdx.y;

 int thread_id = threadIdx.x;
 int num_threads = blockDim.x;

 int warp_id = thread_id / 32;
 int warps_per_blocktile_row = BW / WW;
 int warps_per_blocktile_col = BH / WH;
 int warp_ver = warp_id / warps_per_blocktile_row; // which warptile in the blocktile along the vertical
 int warp_hor = warp_id % warps_per_blocktile_row; // which warptile in the blocktile along the horizontal

 int thread_in_warp = thread_id % 32;
 int threads_per_warptile_row = WW/TW;
 int threads_per_warptile_col = WH/TH;
 int thread_ver = thread_in_warp / (threads_per_warptile_row/w_hor);//thread_in_warp / threads_per_warptile_row; //which threadtile in warptile along the vertical
 int thread_hor = thread_in_warp % (threads_per_warptile_row/w_hor);//thread_in_warp % threads_per_warptile_row;  //which threadtile in warptile along the horizontal

 //top left index of c where the block tile begins - same for 0th BK tile of A & B
  int c_i = block_row * BH; 
  int c_j = block_col * BW;

  int threads_per_BK = BK / TW;
  int A_thread_row = thread_id / threads_per_BK;
  int A_thread_col = 4 * (thread_id % threads_per_BK);
  
  int a_i = c_i + A_thread_row; 
  int a_j = A_thread_col;
  int A_vertical_stride = num_threads / threads_per_BK;

  
  int threads_per_BW = BW/4;
  int B_thread_row = thread_id / threads_per_BW;
  int B_thread_col = 4 * (thread_id % threads_per_BW);

  int b_i = B_thread_row; 
  int b_j = c_j + B_thread_col;
  int B_vertical_stride = num_threads / threads_per_BW;

  
        // which warptile in blocktile does thread belong to
      int warptile_row_in_blocktile = warp_ver*WH;
      // what rows in the warptile does the thread need
      int threadtile_row_in_warptile = thread_ver * TH;

      int a_start = warptile_row_in_blocktile + threadtile_row_in_warptile;
      int a_stride = TH*threads_per_warptile_col/w_vert; //TH * (32/threads_per_warptile_row);

      int warptile_col_in_blocktile = warp_hor * WW;
      int threadtile_col_in_warptile = thread_hor * TW;

      int b_start = warptile_col_in_blocktile + threadtile_col_in_warptile;
      int b_stride = TW * threads_per_warptile_row/w_hor;//TW * (32/threads_per_warptile_row);

 for(int k = 0; k < n; k+=BK){
    // for each BK, load from GMem into SMem and perform compute

    // A : columns change with BK
    //  there are fewer threads in block than elements to load - need to stride vertically
    //  reinterpret cast ensures alignment 
    for(int v=0; v+A_vertical_stride <= BH; v+= A_vertical_stride){
      float4 a_vec = *(reinterpret_cast<float4*>(&A[(a_i+v)*n + a_j + k]));
      // //store transposed A in SMem
      a[A_thread_col * (BH) + A_thread_row + v] = a_vec.x;
      a[(A_thread_col +1) * (BH) + A_thread_row + v] = a_vec.y;
      a[(A_thread_col +2) * (BH) + A_thread_row + v] = a_vec.z;
      a[(A_thread_col +3) * (BH) + A_thread_row + v] = a_vec.w;
      
    }
    
    // B : rows change with BK
    //  there are fewer threads in block than elements to load - need to stride vertically
    //  reinterpret cast ensures alignment 
    for(int v=0; v < BK; v+= B_vertical_stride){
      float4 b_vec = *(reinterpret_cast<float4*>(&B[(b_i+v +k)*n + b_j]));
      // store B in row major in SMem
      b[(B_thread_row + v) * BW + B_thread_col] = b_vec.x;
      b[(B_thread_row + v) * BW + B_thread_col + 1] = b_vec.y;
      b[(B_thread_row + v) * BW + B_thread_col + 2] = b_vec.z;
      b[(B_thread_row + v) * BW + B_thread_col + 3] = b_vec.w;
    }

     __syncthreads();

    // read from SMem (intra warp broadcast) into registers (intra thread reuse during compute)
    
    //determine starting index in warptile then stride based on w_vert & w_hor
    // to populate reg_a, we read columns of a (rows of A), stride based on w_vert

      //Reading from a with i as innermost loop requires more alloctiong to reg_a than available registers
      // Making i the outermost loop and doing the compute after loading each element of all rows in all thread tiles fits in registers
      // loop over w_vert
      for(int i = 0; i < BK; i++){
        //iterate over each element in each row for all vertical threadtiles

        for(int t = 0; t < TH; t++){
          // loop over all TH rows starting from a_start, load from cols in a since SMem is transposed
          
          for(int v = 0; v < w_vert; v++){
            //loop over all thread tiles
            reg_a[v*TH + t] = a[(BH) * i + a_start + t + v*a_stride]; // intra warp broadcast (for threads reading same row)
          }
        }

        // to populate reg_b, we read rows of b (rows of B), stride based on w_hor
        for(int t = 0; t < TW; t++){
          for(int v = 0; v < w_hor; v++){;
            reg_b[v*TW + t] = b[BW * i + b_start + t + v*b_stride];
            //each row of reg_b stores a row of values from 1 thread tile. all rows map to all horizonta threadtiles in warptile
          }
        }

        // compute mat mul, accumulate into c_out
        for(int c = 0; c < w_hor; c++){
          for(int r = 0; r < w_vert; r++){
            for(int tr = 0; tr < TH; tr++){
              for(int tc = 0; tc < TW; tc++){
                c_out[(r*TH + tr)*(w_hor*TW) + (c*TW + tc)] += reg_a[r*TH+tr] * reg_b[c*TW+tc];
              }
            }
          }
        }
      
      }    
     __syncthreads();
 }

 //write results from registers to GMem
 for(int c = 0; c < w_hor; c++){
  for(int r = 0; r < w_vert; r++){
    for(int tr = 0; tr < TH; tr++){
      //TW=4
      int out_row = block_row*BH + warp_ver*WH + thread_ver*TH + r*(a_stride) + tr; 
      int out_col = block_col*BW + warp_hor*WW + thread_hor*TW + c*(b_stride); 

      float4* out_vec = reinterpret_cast<float4*>(&C[out_row*n + out_col]);
      
      *out_vec = make_float4(
        c_out[(r*TH + tr)*(w_hor*TW) + (c*TW)], 
        c_out[(r*TH + tr)*(w_hor*TW) + (c*TW + 1)], 
        c_out[(r*TH + tr)*(w_hor*TW) + (c*TW + 2)], 
        c_out[(r*TH + tr)*(w_hor*TW) + (c*TW + 3)]);
    }
  }
}

}


int main(){
  cudaSetDevice(1); //run on gpu 1

  float *A; float* devA;
  float *B; float* devB;
  float *C; float* devC;
  float *truth;

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float duration;

  for(int n : sizes){
    //Allocate buffers on host and device
    A = (float*) malloc(n * n * sizeof(float));
    B = (float*) malloc(n * n * sizeof(float));
    C = (float*) malloc(n * n * sizeof(float));
    truth = (float*) malloc(n * n * sizeof(float));

    cudaError_t err = cudaMalloc(&devA, n * n * sizeof(float));
    if (err != cudaSuccess){
       cout<<"Dev Memory not allocated"<<endl;
       exit(-1);
    }
    err = cudaMalloc(&devB, n * n * sizeof(float));
    if (err != cudaSuccess){
       cout<<"Dev Memory not allocated"<<endl;
       exit(-1);
    }
    err = cudaMalloc(&devC, n * n * sizeof(float));
    if (err != cudaSuccess){
       cout<<"Dev Memory not allocated"<<endl;
       exit(-1);
    }

    //initialize A, B, C, truth
    init_mats(A, B, C, truth, n);

    //run naive mm to get ground truth - too slow for large n so we use CUBLAS output as ground truth
    // naive_mm(A, B, truth, n);

    cudaMemcpy(devA, A, n*n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, n*n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devC, C, n*n * sizeof(float), cudaMemcpyHostToDevice);

    //run cublas
    cout <<"Running CUBLAS once to get ground truth\n";
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    
    float alpha = 1.0f, beta = 1.0f;
    cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, devB, n, devA, n,&beta,devC, n);
    cudaMemcpy(truth, devC, n*n * sizeof(float), cudaMemcpyDeviceToHost);
    // print_mat(truth, n, "Cublas");
    // save_matrix(truth, n, n, "cublas.txt");

    // cout <<"Checking CUBLAS correctness ....\n";
    // check correctness
    // check_correctness(truth, C, n);

    // benchmark cublas
    cout <<"Benchmarking CUBLAS ....\n";
    cudaEventRecord(start);
    for(int r = 0; r < runs; r++){
      cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, devB, n, devA, n,&beta,devC, n);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&duration, start, end);

    duration /= runs;
    duration /= 1000;

    cout<<"CUBLAS Performance for:\n";
    cout << "\t n:" << n <<"\n";
    cout << "\t Time: " << duration <<" s\n";
    cout << "\t GFLOPs: " << (2*(double)n*n*n*1e-9) / duration <<" GFLOPs/s \n";

    
    //run my kernel
    cout <<"Checking my kernel....\n";
    reset_c(C, n);
    cudaMemset(devC, 0, n * n * sizeof(float));

    dim3 grid(n/BW, n/BH, 1);
    dim3 block(threads_per_block, 1);
    
    kernel<<<grid, block>>>(devA, devB, devC, n);
    cudaMemcpy(C, devC, n*n * sizeof(float), cudaMemcpyDeviceToHost);
    // print_mat(C, n, "Mine");
    // save_matrix(C, n, n, "mine.txt");
    check_correctness(truth, C, n);

    // benchmark my kernel
    duration = 0;
    cout <<"Benchmarking my kernel....\n";
    cudaEventRecord(start);
    for(int r = 0; r < runs; r++){
      kernel<<<grid, block>>>(devA, devB, devC, n);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&duration, start, end);

    duration /= runs;
    duration /= 1000;

    cout<<"Kernel Performance for:\n";
    cout << "\t n:" << n <<"\n";
    cout << "\t Time: " << duration <<" s\n";
    cout << "\t GFLOPs: " << (2*(double)n*n*n*1e-9) / duration <<" GFLOPs/s \n";

    //dealloc
    free(A);
    free(B);
    free(C);
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
  }
  


  return 0;
}
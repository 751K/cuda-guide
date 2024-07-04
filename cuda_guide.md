# CUDA 入门

Owner: 柒柒在笔记
Tags: 技术, 笔记
Created time: November 16, 2023 10:35 PM

# 1. CUDA 基础

## 1.1 Kernel 函数的声明与调用

CUDA C++ 通过称为`kernel`的 函数来实现调用GPU。要创建这样的内核函数，我们用一个特殊的关键字 `__global__` 来声明它。当我们想运行这个内核时，我们不是用普通的方式调用它，而是用一种特殊的语法 `<<<...>>>` 来告诉CUDA我们想要同时运行多少个线程。

在这些内核中，每个线程都能知道自己是第几个线程，通过一个内置的变量 `threadIdx`。这就像每个线程都有一个编号，它们可以用这个编号来区分自己要做的工作。

```cpp
// 定义内核
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x; // 线程编号
    C[i] = A[i] + B[i];  // 计算加法
}

int main()
{
    ...
    // 用N个线程调用内核
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
```

这里，执行 VecAdd() 的 N 个线程中的每一个线程都会执行一次加法。

## 1.2 Thread、Blocks、Grid

在CUDA编程中，我们可以把任务分成很多小块，让显卡上的许多小处理器同时工作，这样可以大大加快计算速度。为了组织这些任务，我们使用了两个概念：线程（threads）和线程块（thread blocks）。

### 线程（Threads）

你可以把每个线程想象成一个小工人，每个人都在做一件小任务。比如，如果我们要把两个大列表里的数字逐个相加，我们可以让每个线程负责加一对数字。

### 线程块（Thread Blocks）

线程块就像是一个团队，里面有很多线程（小工人）。CUDA 允许我们把线程组织成一维、二维或三维的形状，这样可以更方便地处理像列表（一维）、矩阵（二维）或空间数据（三维）这样的结构。Thread Blocks 在 CUDA 编程中确实是一个抽象概念，它并不直接对应于GPU硬件的任何物理部件，其作为组织线程的一种方式，帮助开发者更高效地利用GPU进行并行计算。它们的设计主要是为了方便程序员根据计算任务的特点来组织和管理线程，以及优化程序的执行性能。

线程块是抽象概念，但与SM存在一定的映射关系

**线程块的主要特点包括：**

- 执行上的独立性：一个线程块内的线程可以利用共享内存（Shared Memory）和同步操作来进行通信和协作，但不同线程块之间是相互独立的。这意味着，线程块是 CUDA 并行计算中的一个基本的独立执行单元。
- 逻辑上的映射：尽管线程块是逻辑上的概念，但 CUDA 运行时会将这些线程块映射到GPU上的物理处理核心（Streaming Multiprocessors, SMs）上执行。GPU的调度器负责这种映射，以及管理线程块的执行顺序和资源分配。
- 资源共享与限制：线程块内的线程可以共享一定量的资源，如共享内存。这种资源共享使得线程之间的数据交换和同步变得高效。同时，由于每个SM的资源有限，这也限制了每个线程块和每个SM上可以并行执行的线程块数量。

### 线程的索引 (Thread Index)

每个线程在它的块中都有一个特定的位置或编号，我们称之为线程的索引。这个索引可以是一维的、二维的或三维的，取决于线程块的形状。通过这个索引，线程就知道它负责处理数据的哪一部分。

一维线程块

```cpp
__global__ void addVectors(int *a, int *b, int *c, int N) {
    int index = threadIdx.x;
    if (index < N) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int N = 256; // 假设我们有256个元素
    int blockSize = 256; // 线程块的大小
    addVectors<<<1, blockSize>>>(a, b, c, N);
}
```

二维线程块

```cpp
__global__ void matrixAdd(float *A, float *B, float *C, int width) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int index = row * width + col;
    C[index] = A[index] + B[index];
}

int main() {
    ...
    int width = 16; // 假设矩阵宽度为16
    dim3 threadsPerBlock(16, 16); // 16x16的线程块
    matrixAdd<<<1, threadsPerBlock>>>(A, B, C, width);
    ...
}
```

三维线程块

```cpp
__global__ void process3DData(float *data, int width, int height, int depth) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    int index = x + width * (y + height * z);
    // 对data[index]进行处理
}

int main() {
    ...
    int width = 8, height = 8, depth = 8; // 假设三维空间大小为8x8x8
    dim3 threadsPerBlock(8, 8, 8); // 8x8x8的线程块
    process3DData<<<1, threadsPerBlock>>>(data, width, height, depth);
    ...
}
```

### Dim

在上面的例子中，我们使用了名为 `dim3` 的数据类型。在 CUDA 编程中，`dim3`是一个专门用来表示三维空间维度的数据类型。它被广泛用于指定线程块（block）和网格（grid）的大小。`dim3`类型是CUDA提供的一个结构体，可以用来定义一维、二维或三维的空间维度。`dim3`结构体包含三个无符号整数成员变量`x`、`y`和`z`，分别用于表示三维空间中的宽度、高度和深度。

在定义`dim3`类型的变量时，如果不显式初始化，其成员变量`x`、`y`和`z`的默认值分别为1。这意味着，如果你只需要一维或二维的维度，可以只设置`x`（对于一维）或`x`和`y`（对于二维），而不用担心`z`的值。例如：

```cpp
 dim3 threadsPerBlock(16, 16);
 dim3 threadsPerBlock(8, 8, 8); 
```

### Grid

如同 `Thread` 可以被 `Block` 所组织， `Block` 也可以被 `Grid` 所组织，多个 `Block` 同样可以被组织为一维、二维或三维的线程块网格(`grid`)。与 `Block` 相同，`Grid` 中的`Block`使用内置的`blockIdx`变量进行描述。

**一维Grid**

```cpp
__global__ void processArray(int* array, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 对array[idx]进行处理
    }
}

int main() {
    int N = 10000; // 假设数组大小为10000
    int blockSize = 256; // 每个线程块的线程数
    int numBlocks = (N + blockSize - 1) / blockSize; // 计算所需的线程块数量
    processArray<<<numBlocks, blockSize>>>(array, N);
}
```

![https://www.notion.sogrid-of-thread-blocks.png](https://www.notion.sogrid-of-thread-blocks.png)

**二维Grid**

```cpp
__global__ void processMatrix(float* matrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int index = row * width + col;
        // 对matrix[index]进行处理
    }
}

int main() {
    int width = 1024, height = 768; // 假设矩阵大小为1024x768
    dim3 threadsPerBlock(16, 16); // 定义每个线程块的大小
    dim3 numBlocks((width + 15) / 16, (height + 15) / 16); // 计算所需的网格大小
    processMatrix<<<numBlocks, threadsPerBlock>>>(matrix, width, height);
}
```

**三维Grid**

```cpp
__global__ void processVolume(float* volume, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < width && y < height && z < depth) {
        int index = z * width * height + y * width + x;
        // 对volume[index]进行处理
    }
}

int main() {
    int width = 256, height = 256, depth = 256; // 假设体数据大小为256x256x256
    dim3 threadsPerBlock(8, 8, 8); // 定义每个线程块的大小
    dim3 numBlocks((width + 7) / 8, (height + 7) / 8, (depth + 7) / 8); // 计算所需的网格大小
    processVolume<<<numBlocks, threadsPerBlock>>>(volume, width, height, depth);
}
```

通过这些示例，我们可以看到，根据处理的数据维度选择合适的网格维度，可以更自然地映射计算任务，从而更高效地利用GPU进行并行计算

### Share Memory

每一步 Block 可以拥有一片 Share Memory，用于加速代码运行

```cpp
// CUDA核函数
__global__ void sumArray(int* array, int* result, int N) {
    // 声明共享内存
    __shared__ int sharedMemory[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 将全局内存中的数据加载到共享内存中
    if (idx < N) {
        sharedMemory[tid] = array[idx];
    } else {
        sharedMemory[tid] = 0;
    }

    // 同步所有线程，确保所有数据都已加载到共享内存中
    __syncthreads();

    // 使用二进制归约算法进行求和操作
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedMemory[tid] += sharedMemory[tid + stride];
        }

        // 同步所有线程，确保每一步的求和操作都已完成
        __syncthreads();
    }

    // 将每个线程块的求和结果写回到全局内存中
    if (tid == 0) {
        result[blockIdx.x] = sharedMemory[0];
    }
}

int main() {
    int N = 10000; // 假设数组大小为10000
    int blockSize = 256; // 每个线程块的线程数
    int numBlocks = (N + blockSize - 1) / blockSize; // 计算所需的线程块数量

    int* array; // 输入数组
    int* result; // 结果数组

    // 分配和初始化数组...

    // 调用CUDA核函数
    sumArray<<<numBlocks, blockSize>>>(array, result, N);

    // 处理结果数组...

    return 0;
}
```

## 1.3 存储体系结构

CUDA 线程在执行期间可以从多种内存空间中访问数据。每个Thread 拥有自己的 Register，其空间较小，但速度最快（类比CPU中的 L1 缓存）。当寄存器空间不足时，Thread将把数据存储到 调度器分给该 Thread 的 Local Memory 中。

对于一个 Block，可以通过手动开辟Share Memory 空间来为 Block 块分配一个共享内存，其读写速度相对 Local Memory 较快（因为是在硬件层面进行设计实现的，类似于 L2 缓存）。

| 内存类型 | 生命周期 | 可见性 | 访问速度 |
| --- | --- | --- | --- |
| 寄存器(Register) | 线程 | 线程 | 最快 |
| 局部内存(Local Memory) | 线程 | 线程 | 较慢 |
| 共享内存(Share Memory) | 线程块 | 线程块 | 快 |
| 常量内存(Constant Memory) | 应用程序 | device | 较快 |
| 纹理内存(Texture and Surface Memory) | 应用程序 | device | 较快 |
| 全局内存(Global Memory) | 应用程序 | device | 慢 |

![https://www.notion.somemory-hierarchy.png](https://www.notion.somemory-hierarchy.png)

- 寄存器是最快的内存类型，每个线程都有自己的寄存器。
- 当寄存器不足时，编译器会将数据溢出到局部内存。局部内存的访问速度较慢，因为它位于device内存中。
- 共享内存是每个SM上的一块低延迟内存，可以被同一线程块内的所有线程访问。共享内存的访问速度快于全局内存和常量内存。
- 常量内存是device内存的一部分，用于存储在核函数执行期间不会改变的数据。常量内存有硬件缓存，因此读取速度较快，但写入速度较慢。
- 纹理内存是device内存的一部分，用于存储和查找纹理。纹理内存有硬件缓存，因此读取速度较快，但写入速度较慢。
- 全局内存是device内存的一部分，可以被所有线程访问。全局内存的访问速度较慢，但容量最大。

![Untitled](CUDA%20%E5%85%A5%E9%97%A8%20f85cae70ec504520aeb162bc10134a16/Untitled.png)

## 1.4 异构编程

在CUDA编程中，"Host"和"Device"是两个关键的概念。

- Host：通常指的是CPU与其主机内存（RAM）
- Device：指的是CUDAdevice，也就是GPU及其内存（VRAM）

在 CUDA 程序中，我们通常在Host上执行一些串行的任务（如I/O操作、内存分配等），并在Device上执行并行计算任务。这种模型被称为主机-device模型（Host-Device Model）。

这种模型的一个关键概念是数据传输。因为Host和Device有各自独立的内存空间，所以在执行GPU计算之前，我们需要将数据从Host内存复制到Device内存。同样，计算结果也需要从Device内存复制回Host内存。这种数据传输通过 CUDA 的内存管理函数（如`cudaMemcpy`）来完成。

以下是一个简单的CUDA程序示例，展示了Host和Device的使用：

```cpp
// CUDA核函数
__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}

int main() {
    int a, b, c; // Host变量
    int *d_a, *d_b, *d_c; // Device变量

    // 分配Device内存
    cudaMalloc((void **)&d_a, sizeof(int));
    cudaMalloc((void **)&d_b, sizeof(int));
    cudaMalloc((void **)&d_c, sizeof(int));

    // 初始化Host变量
    a = 2;
    b = 7;

    // 将输入数据复制到Device
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    // 在GPU上启动add()核函数
    add<<<1,1>>>(d_a, d_b, d_c);

    // 将结果复制回Host
    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    // 清理
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}
```

### 统一内存（Unified Memory）

在 CUDA 编程中，统一内存（Unified Memory）是一种特殊的内存管理机制，它为主机（CPU）和device（GPU）提供了一个共享的内存空间。这种内存被称为托管内存（Managed Memory）。

托管内存的主要特点是，它可以被系统中的所有CPU和GPU访问，构建了一个单一的、连贯的内存映像。这意味着你不再需要手动管理主机和device之间的数据传输，CUDA 运行时系统会自动处理这些事情。

当你访问托管内存中的数据时，CUDA 运行时系统会自动检查数据是否在当前device的内存中。如果数据不在当前device内存中，CUDA 会自动将数据从其他device（或主机）内存复制到当前device内存，这个过程被称为数据迁移（Data Migration）。

此外，统一内存还支持device内存的超额订阅（Device Memory Oversubscription）。这意味着你的程序可以请求超过GPU物理内存大小的内存空间，CUDA运行时系统会自动管理内存，确保当前需要的数据总是在device内存中。

总的来说，统一内存极大地简化了 CUDA 编程，使得程序员可以更专注于并行算法的设计，而不是内存管理。然而，在实际编程中，为获得最高的运行效率，CUDA 编程中依然需要对内存进行管理。

## 1.5 数据流

在 CUDA 编程中，流是一个非常重要的概念。流（Stream）是一系列按顺序执行的操作（包括核函数调用、内存传输等）的序列。在同一个流中，操作是按照它们被调度的顺序依次执行的。

在 CUDA 中，每个流都表示一组按顺序执行的 CUDA 命令。默认情况下，所有的 CUDA 操作都在一个默认的流中执行，这意味着所有的操作都是串行执行的。然而，CUDA也支持用户创建多个流，这样就可以在GPU上并行执行多个操作。

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 在stream1中执行核函数
kernel<<<blocks, threads, 0, stream1>>>(...);

// 在stream2中执行内存传输
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream2);

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

### 1.5.1 Default Stream

当代码中没有显式去创建一个Stream的时候，编译器即创建一个**Default Stream。Default Stream**（默认流）是一个特殊的Stream，它用于执行那些没有显式指定Stream参数的内核启动和内存拷贝操作。默认流的行为对并发执行和同步有重要影响。

**默认流的类型**

1. Legacy Default Stream（传统默认流）：
    - 所有主机线程共享同一个默认流，称为`NULL Stream`。
    - 在同一个device上运行的所有主机线程中的命令都会在此流上同步执行。
    - 适合需要严格顺序执行的场景。
    - 如果不指定编译标志，CUDA代码编译时默认使用传统默认流模式
2. Per-Thread Default Stream（每线程默认流）：
    - 每个主机线程都有自己的默认流。
    - 不同主机线程的默认流之间可以并发执行，适合并行执行的多线程程序。
    - 可以通过编译标志`-default-stream per-thread`或者定义宏`CUDA_API_PER_THREAD_DEFAULT_STREAM`来启用这种模式。

例如，在下面的代码中，即隐式创建了一个 Default Stream

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void MyKernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] += 1.0f;
}

int main() {
    const int size = 1024;
    float* h_data;
    cudaMallocHost(&h_data, size * sizeof(float)); // 锁页内存
    float* d_data;
    cudaMalloc(&d_data, size * sizeof(float));

    // 使用默认流执行内存拷贝和内核启动
    cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);
    MyKernel<<<size / 256, 256>>>(d_data);
    cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFreeHost(h_data);

    return 0;
}

```

### 1.5.2 Stream 的显式同步

1. `cudaDeviceSynchronize()`

用于主机等待device上所有任务完成。它会阻塞主机线程，直到device上所有主机线程发出的所有命令都完成。

**使用场景**

- 当你需要确保device上所有正在执行的任务都已完成时，使用此函数。
- 适用于需要全局同步的情况。

2. `cudaStreamSynchronize(cudaStream_t stream)`

用于主机等待特定Stream中的所有任务完成。它会阻塞主机线程，直到指定Stream中的所有命令都执行完毕。

**使用场景**：

- 当你只需要同步特定Stream中的任务时，使用此函数。
- 适用于需要局部同步的情况，而其他Stream可以继续执行。

**示例**：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

// 发出一些命令到stream
cudaMemcpyAsync(..., stream);
MyKernel<<<grid, block, 0, stream>>>(...);

// 同步特定的stream
cudaStreamSynchronize(stream);
```

3. `cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)`

用于在指定的事件完成之前，延迟特定Stream中的命令执行。它可以让Stream等待某个事件的完成，然后再执行接下来的命令。

**使用场景**：

- 当你需要在一个Stream中的任务开始执行之前，等待另一个Stream中的某个事件完成时，使用此函数。
- 适用于需要跨Stream同步的情况。

**示例**：`stream2`等待`event`完成，然后再执行后续命令

```cpp
cudaEvent_t event;
cudaEventCreate(&event);
cudaEventRecord(event, stream1);

// 让stream2等待event完成
cudaStreamWaitEvent(stream2, event, 0);

// 在stream2中发出一些命令
cudaMemcpyAsync(..., stream2);
MyKernel<<<grid, block, 0, stream2>>>(...);
```

4. `cudaStreamQuery(cudaStream_t stream)`

用于查询特定Stream中的所有命令是否已经完成。它不会阻塞主机线程，而是立即返回一个状态。

**使用场景**：

- 当你需要检查特定Stream中的任务是否完成，而不希望主机线程阻塞时，使用此函数。

**返回值**：

- 如果Stream中的所有任务已经完成，返回`cudaSuccess`。
- 如果还有未完成的任务，返回`cudaErrorNotReady`。

**示例**：

```cpp
cudaError_t status = cudaStreamQuery(stream);

if (status == cudaSuccess) {
    // Stream中的所有任务已经完成
} else if (status == cudaErrorNotReady) {
    // Stream中还有未完成的任务
}
```

**解释**：

- 这段代码会立即检查`stream`中的任务状态，而不会阻塞主机线程。

| 函数名 | 功能说明 | 使用场景 |
| --- | --- | --- |
| cudaDeviceSynchronize() | 等待所有主机线程的所有Stream中的所有命令完成 | 需要全局同步所有任务 |
| cudaStreamSynchronize(cudaStream_t stream) | 等待特定Stream中的所有命令完成 | 需要同步特定Stream中的任务，其他Stream可继续执行 |
| cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) | 在指定事件完成前，延迟特定Stream中的命令执行 | 需要在一个Stream等待另一个Stream的事件完成时使用 |
| cudaStreamQuery(cudaStream_t stream) | 查询特定Stream中的所有命令是否已完成，不阻塞主机线程 | 需要非阻塞地检查Stream中的任务是否完成 |

### 1.5.3 Stream 的隐式同步

隐式同步是指在某些情况下，CUDA会自动对不同Stream中的命令进行同步，而无需显式调用同步函数。

- **锁页主机内存分配**：
    - 当主机分配锁页内存（页面锁定内存）时，会触发隐式同步。
- **device内存分配**：
    - 当device分配内存时，会触发隐式同步。
- **device内存设置**：
    - 当设置device内存（如使用`cudaMemset`）时，会触发隐式同步。
- **两个地址之间的内存拷贝到同一device内存**：
    - 如果在同一device内存上进行两个地址之间的内存拷贝，会触发隐式同步。
- **对NULL Stream的任何CUDA命令**：
    - 对NULL Stream（默认Stream）的任何CUDA命令都会触发隐式同步。
- **在计算能力3.x和计算能力7.x中描述的L1/共享内存配置之间的切换**：
    - 切换L1和共享内存配置会触发隐式同步

给出实现隐式同步的代码

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void MyKernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] += 1.0f;
}

int main() {
    const int size = 1024;
    float* h_data;
    cudaMallocHost(&h_data, size * sizeof(float)); // 锁页内存分配，可能触发隐式同步
    float* d_data;
    cudaMalloc(&d_data, size * sizeof(float)); // device内存分配，可能触发隐式同步

    // 创建两个Stream
    cudaStream_t stream[2];
    for (int i = 0; i < 2; ++i)
        cudaStreamCreate(&stream[i]);

    // 在Stream 0中执行内存拷贝和内核启动
    cudaMemcpyAsync(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice, stream[0]);
    MyKernel<<<size / 256, 256, 0, stream[0]>>>(d_data);
    cudaMemcpyAsync(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost, stream[0]);

    // 在Stream 1中执行另一个内存拷贝和内核启动
    cudaMemcpyAsync(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice, stream[1]);
    MyKernel<<<size / 256, 256, 0, stream[1]>>>(d_data);
    cudaMemcpyAsync(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost, stream[1]);

    // 等待Stream 0完成
    cudaStreamSynchronize(stream[0]);

    // 等待Stream 1完成
    cudaStreamSynchronize(stream[1]);

    // 销毁Stream
    for (int i = 0; i < 2; ++i)
        cudaStreamDestroy(stream[i]);

    cudaFree(d_data);
    cudaFreeHost(h_data);

    return 0;
}
```

### 1.5.4 主机回调函数

CUDA运行时提供了一种机制，可以在Stream的任何位置插入CPU函数调用。这是通过`cudaLaunchHostFunc()`实现的。`cudaLaunchHostFunc()`允许在指定的Stream中插入一个回调函数。在Stream中的所有命令完成后，这个回调函数会在主机上执行。

- **定义回调函数**：
    - 回调函数需要符合特定的签名，例如下面的`MyCallback`函数。
    - 回调函数将在Stream中的所有命令完成后执行。
- **在Stream中插入回调函数**：
    - 使用`cudaLaunchHostFunc()`将回调函数插入到指定的Stream中。
    - 在回调函数完成之前，向Stream中发射的其他命令不会开始执行。

下面给出一个简单的示例

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void MyKernel(float* data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] += 1.0f;
    }
}

// 回调函数
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void* data) {
    printf("Inside callback %zu\n", reinterpret_cast<size_t>(data));
}

int main() {
    const int size = 1024 * 1024; // 1MB 数据大小
    const int numStreams = 2;
    float* h_data[numStreams];
    float* d_data_in[numStreams];
    float* d_data_out[numStreams];

    // 分配锁页主机内存和device内存
    for (int i = 0; i < numStreams; ++i) {
        cudaMallocHost(&h_data[i], size * sizeof(float)); // 锁页主机内存
        cudaMalloc(&d_data_in[i], size * sizeof(float)); // device输入内存
        cudaMalloc(&d_data_out[i], size * sizeof(float)); // device输出内存
        // 初始化主机数据
        for (int j = 0; j < size; ++j) {
            h_data[i][j] = static_cast<float>(j);
        }
    }

    // 创建多个Stream
    cudaStream_t streams[numStreams];
    for (auto & stream : streams) {
        cudaStreamCreate(&stream);
    }

    // 启动异步内存复制、内核执行并插入回调函数
    for (size_t i = 0; i < numStreams; ++i) {
        // 异步内存复制：从主机到device
        cudaMemcpyAsync(d_data_in[i], h_data[i], size * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        // 启动内核
        MyKernel<<<(size + 255) / 256, 256, 0, streams[i]>>>(d_data_in[i], size);
        // 异步内存复制：从device到主机
        cudaMemcpyAsync(h_data[i], d_data_in[i], size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
        // 插入回调函数
        cudaLaunchHostFunc(streams[i], reinterpret_cast<cudaHostFn_t>(MyCallback), (void*)i);
    }

    // 同步Stream
    for (auto & stream : streams) {
        cudaStreamSynchronize(stream);
    }

    // 验证结果
    bool success = true;
    for (auto & i : h_data) {
        for (int j = 0; j < size; ++j) {
            if (i[j] != static_cast<float>(j) + 1.0f) {
                success = false;
                break;
            }
        }
    }

    if (success) {
        std::cout << "Succeed!" << std::endl;
    } else {
        std::cout << "Failed!" << std::endl;
    }

    // 清理资源
    for (int i = 0; i < numStreams; ++i) {
        cudaFreeHost(h_data[i]);
        cudaFree(d_data_in[i]);
        cudaFree(d_data_out[i]);
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}

```

### 1.5.5 Stream Priorities（流优先级）

在创建Stream时，可以指定该Stream的相对优先级。使用优先级可以控制不同Stream中的任务执行顺序。高优先级的Stream中的任务会优先于低优先级的Stream中的任务执行。

1. 获取优先级范围
    1. 使用`cudaDeviceGetStreamPriorityRange()`函数可以获取当前device支持的Stream优先级范围。优先级按从高到低排序，即较高的数值表示较高的优先级，较低的数值表示较低的优先级。
2. 创建具有指定优先级的Stream
    1. 使用`cudaStreamCreateWithPriority()`函数可以创建具有指定优先级的Stream。在创建Stream时，可以指定优先级参数，从而控制该Stream的相对优先级。

下面给出一份基础的代码示例

```cpp
#include <cuda_runtime.h>
#include <iostream>

int main() {
    // 获取当前device支持的流优先级范围
    int priority_high, priority_low;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    std::cout << "Highest priority: " << priority_high << ", Lowest priority: " << priority_low << std::endl;

    // 创建具有最高优先级的Stream
    cudaStream_t st_high;
    cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, priority_high);

    // 创建具有最低优先级的Stream
    cudaStream_t st_low;
    cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, priority_low);

    // 示例操作：在不同优先级的Stream中执行任务
    const int size = 1024;
    float* d_data_high;
    float* d_data_low;
    cudaMalloc(&d_data_high, size * sizeof(float));
    cudaMalloc(&d_data_low, size * sizeof(float));

    // 异步内存操作和内核启动在高优先级Stream中
    cudaMemcpyAsync(d_data_high, d_data_high, size * sizeof(float), cudaMemcpyDeviceToDevice, st_high);
    // 这里可以添加内核执行等其他操作

    // 异步内存操作和内核启动在低优先级Stream中
    cudaMemcpyAsync(d_data_low, d_data_low, size * sizeof(float), cudaMemcpyDeviceToDevice, st_low);
    // 这里可以添加内核执行等其他操作

    // 同步Stream
    cudaStreamSynchronize(st_high);
    cudaStreamSynchronize(st_low);

    // 清理资源
    cudaFree(d_data_high);
    cudaFree(d_data_low);
    cudaStreamDestroy(st_high);
    cudaStreamDestroy(st_low);

    return 0;
}
```

## 1.6 同步与异步

- 同步：`host` 向 `device` 提交任务，在同步的情况下，`host` 将会阻塞，知道 `device` 将所提交任务完成，并将控制权交回 `host`，然后会继续执行主机的程序；
- 异步：`host` 向 `device` 提交任务后， `device` 开始执行任务，并立刻将控制权交回host ，所以 `host` 将不会阻塞，而是直接继续执行 `host` 的程序，即在异步的情况下，`host` 不会等待 `device` 执行任务完成；

（这里可能需要有一定的硬件电路基础，理解阻塞与非阻塞之间的区别）

显然，同步操作较为简洁，而异步操作则减少了在 `host` 与 `device` 之间的性能损失。与异步相对应的概念即为异步操作。异步操作的核心在于，进行操作的时候，不会阻碍host的操作，在device处理数据的时候，host 并不需要等待 device 结束操作再进行其他操作

如何在 Cuda 编程中使用异步操作去提升效率：结合2.5 数据流的概念

下面是一个示例：

```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 512
#define STREAM_COUNT 4

// CUDA 核函数，用于将两个数组的元素逐一相加
__global__ void add(int *a, int *b, int *c) {
    int index = threadIdx.x;
    c[index] = a[index] + b[index];
}

// 生成随机整数数组
void random_ints(int *a, int size) {
    for (int i = 0; i < size; i++) {
        a[i] = rand() % 100;
    }
}

int main() {
    int *a[STREAM_COUNT], *b[STREAM_COUNT], *c[STREAM_COUNT]; // host 上的数组指针
    int *d_a[STREAM_COUNT], *d_b[STREAM_COUNT], *d_c[STREAM_COUNT]; // device 上的数组指针
    int size = N * sizeof(int);

    cudaStream_t streams[STREAM_COUNT];

    // 为每个 stream 分配内存并生成随机数据
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaMalloc((void **)&d_a[i], size);
        cudaMalloc((void **)&d_b[i], size);
        cudaMalloc((void **)&d_c[i], size);

        a[i] = (int *)malloc(size); random_ints(a[i], N);
        b[i] = (int *)malloc(size); random_ints(b[i], N);
        c[i] = (int *)malloc(size);

        cudaStreamCreate(&streams[i]);
    }

    // 异步地将 host 上的数据复制到 device，并启动 CUDA 核函数
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaMemcpyAsync(d_a[i], a[i], size, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_b[i], b[i], size, cudaMemcpyHostToDevice, streams[i]);
        
        add<<<1, N, 0, streams[i]>>>(d_a[i], d_b[i], d_c[i]);
        cudaMemcpyAsync(c[i], d_c[i], size, cudaMemcpyDeviceToHost, streams[i]);
    }

    // 同步所有流，确保所有异步操作完成
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // 清理内存
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaFree(d_a[i]); cudaFree(d_b[i]); cudaFree(d_c[i]);
        free(a[i]); free(b[i]); free(c[i]);
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}

```

- **CUDA流的创建和使用**：
    - 创建了4个CUDA流，每个流可以独立执行操作。
    - 为每个流分配内存并生成随机数据。
- **异步内存传输和内核函数调用**：
    - 使用`cudaMemcpyAsync`将主机上的数据异步地复制到device。
    - 使用多个流并行执行不同的任务，每个流中的操作是顺序执行的，但不同流之间的操作是并行执行的。
- **流的同步**：
    - 使用`cudaStreamSynchronize`确保每个流中的操作完成后，主机才继续执行。

### 每个流中的顺序（以第一个流为例）

对于每个流（以第一个流为例），操作的顺序如下：

1. `cudaMemcpyAsync(d_a[0], a[0], size, cudaMemcpyHostToDevice, streams[0]);` 开始
2. `cudaMemcpyAsync(d_a[0], a[0], size, cudaMemcpyHostToDevice, streams[0]);` 结束
3. `cudaMemcpyAsync(d_b[0], b[0], size, cudaMemcpyHostToDevice, streams[0]);` 开始
4. `cudaMemcpyAsync(d_b[0], b[0], size, cudaMemcpyHostToDevice, streams[0]);` 结束
5. `add<<<1, N, 0, streams[0]>>>(d_a[0], d_b[0], d_c[0]);` 开始
6. `add<<<1, N, 0, streams[0]>>>(d_a[0], d_b[0], d_c[0]);` 结束
7. `cudaMemcpyAsync(c[0], d_c[0], size, cudaMemcpyDeviceToHost, streams[0]);` 开始
8. `cudaMemcpyAsync(c[0], d_c[0], size, cudaMemcpyDeviceToHost, streams[0]);` 结束

### 不同流之间的并行执行

多个流（例如流0和流1）之间的操作是并行执行的。例如：

- 流0和流1同时执行`cudaMemcpyAsync(d_a, a, size, cudaMemcpyHostToDevice, stream);`
- 流0和流1同时执行`cudaMemcpyAsync(d_b, b, size, cudaMemcpyHostToDevice, stream);`
- 流0和流1同时执行`add<<<1, N, 0, stream>>>(d_a, d_b, d_c);`
- 流0和流1同时执行`cudaMemcpyAsync(c, d_c, size, cudaMemcpyDeviceToHost, stream);`

### 异步（Asynchronous Operations ）后的同步管理

可以看到，在上面的代码中，使用了

```cpp
cudaStreamSynchronize(streams[i]);
```

去同步数据流。它确保流中的所有操作都完成后，主机代码才继续执行。

```cpp
for (int i = 0; i < STREAM_COUNT; i++) {
    cudaMemcpyAsync(d_a[i], a[i], size, cudaMemcpyHostToDevice, streams[i]);
    cudaMemcpyAsync(d_b[i], b[i], size, cudaMemcpyHostToDevice, streams[i]);
    add<<<1, N, 0, streams[i]>>>(d_a[i], d_b[i], d_c[i]);
    cudaMemcpyAsync(c[i], d_c[i], size, cudaMemcpyDeviceToHost, streams[i]);
}

for (int i = 0; i < STREAM_COUNT; i++) {
    cudaStreamSynchronize(streams[i]);
}

```

在上述循环中，我们对每个流（例如`streams[i]`）都执行了异步操作。这些操作将按照它们在流中的顺序执行，但不会阻塞主机线程。

当我们调用`cudaStreamSynchronize(streams[i])`时，程序会等待指定流中的所有操作完成，然后才继续执行后续代码。这是一个阻塞调用，用来确保所有异步操作完成。

异步操作使用同步对象来同步操作的完成。这样的同步对象可以由用户显式管理（例如，`cuda::memcpy_async`）或在库中隐式管理（例如，`cooperative_groups::memcpy_async`）。不同的同步对象有不同的作用域。

| Thread Scope | 描述 |
| --- | --- |
| cuda::thread_scope::thread_scope_thread | 只有发起异步操作的 CUDA thread 进行同步。 |
| cuda::thread_scope::thread_scope_block | 处于与发起线程相同的 thread block 中的所有或任何 CUDA thread 进行同步。 |
| cuda::thread_scope::thread_scope_device | 处于与发起线程相同的 GPU device 中的所有或任何 CUDA thread 进行同步。 |
| cuda::thread_scope::thread_scope_system | 处于与发起线程相同的 system 中的所有或任何 CUDA 或 CPU thread 进行同步。 |

这些线程作用域是在[CUDA 标准 C++库](https://nvidia.github.io/libcudacxx/extended_api/thread_scopes.html)中作为标准C++的扩展来实现的。

## 1.7 CUDA Context

在 CUDA 编程中，CUDA 上下文（ CUDA  Context）是一个重要的概念，它代表了 CUDA 驱动程序在GPU上执行计算和管理资源的一个操作环境。

- CUDA Context 管理 CUDA 程序所需的所有资源，包括内存分配、内核函数编译、device配置等。
- 每个 CUDA Context 都有自己的独立资源集合，不同上下文之间的资源不能直接共享。
- 上下文保存了CUDA程序执行过程中需要的状态信息，例如当前活动的device、内核函数参数、内存分配状态等
- 每个CUDA上下文与一个特定的GPUdevice关联。程序可以在同一device上创建多个上下文，但这些上下文之间是独立的。

## 1.8 环境管理

### 1.8.1 版本管理

进行CUDA开发时，主要有两个需要关注的版本数值：

- 计算能力（Compute Capability）
    - 设备的一般规格和功能
- CUDA 驱动 API 版本
    - 驱动 API 和运行时支持的功能。

驱动 API 的版本在驱动头文件中定义为 `CUDA_VERSION`。驱动 API 向后兼容，这意味着针对特定版本（例如11.7）的驱动 API 编译的应用程序、插件和库将在后续的设备驱动程序版本（例如11.8）上继续工作。相反的，驱动 API 不向前兼容，这意味着针对特定版本的驱动 API （例如11.8）编译的应用程序、插件和库（包括 CUDA 运行时）在以前的设备驱动程序版本（例如11.7）上将无法工作。

需要注意的是，混合和匹配版本有一些限制：

1. 由于系统上一次只能安装一个版本的 CUDA 驱动程序，因此安装的驱动程序必须与在该系统上运行的任何应用程序、插件或库所构建的最高驱动 API 版本相同或更高。
2. 所有由应用程序使用的插件和库必须使用相同版本的 CUDA Runtime，除非它们静态链接到运行时，在这种情况下，多个版本的运行时可以在同一进程空间中共存。请注意，如果使用 `nvcc` 链接应用程序，默认情况下将使用 CUDA 运行时库的静态版本，并且所有 CUDA 工具包库都静态链接到 CUDA Runtime。
3. 所有由应用程序使用的插件和库必须使用相同版本的任何使用运行时的库（例如 cuFFT、cuBLAS 等），除非静态链接到这些库

设备的计算能力由一个版本号表示，有时也称为其“SM版本”。这个版本号标识了GPU硬件支持的功能，应用程序在运行时使用它来确定当前GPU上可用的硬件功能和/或指令。计算能力由一个主修订号 X 和一个次修订号 Y 组成，表示为 X.Y。

具有相同主修订号的设备属于相同的核心架构。基于不同架构的设备，其主修订号如下：

- NVIDIA Hopper GPU 架构的设备，其主修订号为 9。
- NVIDIA Ampere GPU 架构的设备，其主修订号为 8。
- Volta 架构的设备，其主修订号为 7。
- Pascal 架构的设备，其主修订号为 6。
- Maxwell 架构的设备，其主修订号为 5。
- Kepler 架构的设备，其主修订号为 3。

次修订号对应于核心架构的增量改进，可能包括新功能。

### 1.8.2 计算模式

在运行 Windows Server 2008 及更高版本或 Linux 的 Tesla 解决方案上，可以使用 NVIDIA 的系统管理接口 (nvidia-smi) 将系统中的任何设备设置为以下三种模式之一。

- **默认计算模式**：多个主机线程可以同时使用该设备（在使用运行时 API 时通过调用 `cudaSetDevice()`，或在使用驱动 API 时通过将与设备关联的上下文设为当前上下文）。
- **独占进程计算模式**：在系统中的所有进程中只能在设备上创建一个 CUDA 上下文。该上下文可以在创建它的进程中的任意多个线程中设为当前上下文。
- **禁止计算模式**：不能在设备上创建 CUDA 上下文。

这意味着，特别是当设备 0 处于禁止模式或被其他进程使用的独占进程模式时，使用运行时 API 的主机线程如果没有显式调用 `cudaSetDevice()` 选择设备0，进程可能会与设备 0 以外的设备关联。可以使用 `cudaSetValidDevices()` 从设备的优先列表中设置设备。

此外，对于 Pascal 架构及更高版本（计算能力主修订号为 6 及以上）的设备，支持计算抢占（Compute Preemption）。这允许计算任务在指令级粒度（而不是先前的 Maxwell 和 Kepler GPU 架构中的线程块粒度）上被抢占，这样可以防止运行时间较长的内核应用程序垄断系统或超时。然而，计算抢占会带来上下文切换开销，且在支持计算抢占的设备上自动启用。可以使用 `cudaDeviceGetAttribute()` 函数查询 `cudaDevAttrComputePreemptionSupported` 属性来确定所用设备是否支持计算抢占。希望避免与不同进程相关的上下文切换开销的用户可以通过选择独占进程模式确保 GPU 上只有一个进程在活动。应用程序可以通过检查 `computeMode` 设备属性来查询设备的计算模式

### 1.8.3 计算集群模式

使用 NVIDIA 的系统管理接口 (nvidia-smi)，可以将 Tesla 和 Quadro 系列设备的 Windows 驱动程序置于 TCC（Tesla Compute Cluster）模式。

TCC 模式移除了对任何图形功能的支持。

# 2. 内存管理

## 2.1  内存空间开辟与释放

Cuda 的内存空间主要分为两种

- 线性内存（Linear Memory）
    - 在一个统一的地址空间中分配，可以通过指针相互引用，例如在二叉树或链表中。
- CUDA数组（CUDA Arrays）
    - 不透明的内存布局，优化用于纹理提取，详细内容见纹理和表面内存部分。

### 2.1.1 使用 cudaMalloc() 分配线性内存

`cudaMalloc()`用于分配线性内存，即一维连续的内存块。通常用于分配一维数组。

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void VecAdd(float* A, float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);
    
    // 主机内存分配
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    
    // 初始化主机内存
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // device内存分配
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 将数据从主机复制到devicevice
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 启动内核
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 将结果从devicevice复制回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 释放device内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 释放主机内存
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

```

### 2.1.2 使用 cudaMallocPitch() 分配二维内存

`cudaMallocPitch()`用于分配二维内存，并确保行对齐以优化访问性能。

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void Process2DArray(float* devPtr, size_t pitch, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float* row = (float*)((char*)devPtr + y * pitch);
        row[x] = x + y; 
    }
}

int main() {
    int width = 64, height = 64;
    size_t pitch;
    float* devPtr;

    // 分配二维内存
    cudaMallocPitch(&devPtr, &pitch, width * sizeof(float), height);

    // 启动内核
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    Process2DArray<<<blocksPerGrid, threadsPerBlock>>>(devPtr, pitch, width, height);

    // 释放内存
    cudaFree(devPtr);

    return 0;
}

```

### 2.1.3 使用 cudaMalloc3D() 分配三维内存

`cudaMalloc3D()`用于分配三维内存，并返回一个`cudaPitchedPtr`结构体，其中包含指向内存的指针、行间距和层间距。

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void Process3DArray(cudaPitchedPtr devPitchedPtr, int width, int height, int depth) {
    char* devPtr = devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        char* slice = devPtr + z * slicePitch;
        float* row = (float*)(slice + y * pitch);
        row[x] = x + y + z; // 简单赋值操作
    }
}

int main() {
    int width = 64, height = 64, depth = 64;
    cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
    cudaPitchedPtr devPitchedPtr;

    // 分配三维内存
    cudaMalloc3D(&devPitchedPtr, extent);

    // 启动内核
    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (depth + threadsPerBlock.z - 1) / threadsPerBlock.z);
    Process3DArray<<<blocksPerGrid, threadsPerBlock>>>(devPitchedPtr, width, height, depth);

    // 释放内存
    cudaFree(devPitchedPtr.ptr);

    return 0;
}

```

### 2.1.4 差别

1. **cudaMalloc()**：
    - 分配线性内存，适用于一维数组。
    - 返回一个指向分配内存的指针。
2. **cudaMallocPitch()**：
    - 分配二维内存，并确保行对齐以优化访问性能。
    - 返回指向内存的指针和行间距（pitch）。
3. **cudaMalloc3D()**：
    - 分配三维内存，并返回一个`cudaPitchedPtr`结构体。
    - 结构体包含指向内存的指针、行间距（pitch）和层间距（slicePitch）。

占位符

## 2.2  共享内存 Shared Memory

以一个 (1024,1024) 的矩阵乘法为例：

1. 总体规模：矩阵 A、B、C 的大小都是 1024 x 1024。
2. 线程、块和网格的划分：
    1. 每个线程（thread）：计算结果矩阵 C 中的一个元素。
    2. 每个线程块（block）包含 16 x 16 个线程（thread），即 `BLOCK_SIZE` 为 16。处理结果矩阵 C 中的一个子矩阵（16 x 16 的块）。
    3. 整个网格（grid）：覆盖整个结果矩阵 C。包含 `(1024 / 16) x (1024 / 16)` 个块，即 64 x 64 个块

**共享内存的作用**

共享内存用于存储当前块（block）中所需的子矩阵（子块），从而减少对全局内存的访问，提高计算效率。

- 共享内存大小：每个块的共享内存大小为 16 x 16，分别用于存储 A 和 B 的子块。
- 共享内存存储内容：每个线程块需要加载 A 和 B 的 16 x 16 子块到共享内存中进行计算。

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib> 

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

#define BLOCK_SIZE 16

__global__ void MatMulKernelNoShared(const Matrix A, const Matrix B, Matrix C);
__global__ void MatMulKernelShared(const Matrix A, const Matrix B, Matrix C);

void MatMul(Matrix A, Matrix B, Matrix C, bool useSharedMemory, float &time);
void InitializeMatrix(Matrix &mat, int width, int height);
void RandomizeMatrix(Matrix &mat);
void FreeMatrix(Matrix &mat);

int main() {
    // 定义矩阵维度
    int width = 1024;
    int height = 1024;

    // 分配并初始化矩阵
    Matrix A, B, C;
    InitializeMatrix(A, width, height);
    InitializeMatrix(B, width, height);
    InitializeMatrix(C, width, height);

    // 不使用共享内存的矩阵乘法
    float timeNoShared = 0;
    for (int i = 0; i < 100; ++i) {
        float tempTime;
        RandomizeMatrix(A); // 随机生成矩阵A的数据
        RandomizeMatrix(B); // 随机生成矩阵B的数据
        MatMul(A, B, C, false, tempTime);
        timeNoShared += tempTime;
    }
    std::cout << "without share memory: " << timeNoShared / 100 << " ms" << std::endl;

    // 使用共享内存的矩阵乘法
    float timeShared = 0;
    for (int i = 0; i < 100; ++i) {
        float tempTime;
        RandomizeMatrix(A); // 随机生成矩阵A的数据
        RandomizeMatrix(B); // 随机生成矩阵B的数据
        MatMul(A, B, C, true, tempTime);
        timeShared += tempTime;
    }
    std::cout << "with share memory: " << timeShared / 100 << " ms" << std::endl;

    // 释放矩阵内存
    FreeMatrix(A);
    FreeMatrix(B);
    FreeMatrix(C);

    return 0;
}

void MatMul(Matrix A, Matrix B, Matrix C, bool useSharedMemory, float &time) {
    // 将A和B加载到device内存
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    // 在device内存中分配C
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // 设置线程块和网格维度
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y - 1) / dimBlock.y);

    // 开始计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 调用内核
    if (useSharedMemory)
        MatMulKernelShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    else
        MatMulKernelNoShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // 停止计时
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    // 从device内存读取C
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    // 释放device内存
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

__global__ void MatMulKernelNoShared(Matrix A, Matrix B, Matrix C) {
    // 每个线程计算C的一个元素
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

__global__ void MatMulKernelShared(Matrix A, Matrix B, Matrix C) {
    // 声明共享内存
    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // 计算C中当前元素的位置
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    float Cvalue = 0;

    // 遍历A和B的子块
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // 每个线程将A和B的子块加载到共享内存
        sharedA[ty][tx] = A.elements[row * A.width + (m * BLOCK_SIZE + tx)];
        sharedB[ty][tx] = B.elements[(m * BLOCK_SIZE + ty) * B.width + col];

        // 同步，确保所有线程都加载了数据
        __syncthreads();

        // 计算当前子块的C值
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += sharedA[ty][e] * sharedB[e][tx];

        // 同步，确保所有线程都完成了计算
        __syncthreads();
    }

    // 将结果写入C
    C.elements[row * C.width + col] = Cvalue;
}

void InitializeMatrix(Matrix &mat, int width, int height) {
    mat.width = width;
    mat.height = height;
    mat.elements = (float*)malloc(width * height * sizeof(float));
}

void RandomizeMatrix(Matrix &mat) {
    for (int i = 0; i < mat.width * mat.height; ++i) {
        mat.elements[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void FreeMatrix(Matrix &mat) {
    free(mat.elements);
}

```

运行结果如下：

```cpp
without share memory: 3.84465 ms
with share memory: 2.79083 ms
```

充分说明使用共享内存能有效提升运行速度

## 2.3  线程块簇与分布式共享内存

线程块簇（Thread Block Clusters）与分布式共享内存（Distributed Shared Memory）是CUDA 9.0引入的一个新概念，它允许一组线程块作为一个簇（cluster）来进行协同工作。簇中的线程块可以共享资源，

分布式共享内存是指一个线程块簇中的所有线程块可以访问簇内所有线程块的共享内存。这个分布式共享内存地址空间允许线程跨越线程块边界进行内存访问，使得每个线程块不仅可以访问自身的共享内存，还可以访问同一簇中其他线程块的共享内存。

在传统的CUDA编程模型中，线程块之间的通信和同步是非常有限的。虽然同一个线程块内的线程可以高效地通过共享内存和同步机制进行通信，但不同线程块之间的通信只能通过全局内存，且无法直接同步。这在某些计算任务中，如大规模矩阵运算或复杂的科学计算，可能会导致性能瓶颈。

线程块簇通过引入一个新的分布式共享内存（Distributed Shared Memory）概念，使得簇内的所有线程块可以共享彼此的共享内存，并且可以进行跨线程块的同步，从而极大地提高了程序的并行计算性能和编程灵活性。

下面提供了一个使用线程块簇实现直方图计算的例子：

```cpp
#include <cooperative_groups.h>

// 直方图计算核函数，使用分布式共享内存
__global__ void clusterHist_kernel(int *bins, const int nbins, const int bins_per_block, const int *__restrict__ input, size_t array_size)
{
    extern __shared__ int smem[];  // 声明共享内存
    namespace cg = cooperative_groups;  // 使用cooperative_groups命名空间
    int tid = cg::this_grid().thread_rank();  // 获取网格内线程的唯一ID

    // 初始化线程簇和簇内相关变量
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;
    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
    {
        smem[i] = 0;  // 将共享内存中的直方图bin初始化为0
    }
    // 同步簇内所有线程，确保所有共享内存初始化完毕
    cluster.sync();

    // 遍历输入数据，根据数据值更新直方图bin
    for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
    {
        int ldata = input[i];
        int binid = ldata;
        if (ldata < 0)
            binid = 0;
        else if (ldata >= nbins)
            binid = nbins - 1;
        
        // 计算目标线程块的rank和偏移量
        int dst_block_rank = (int)(binid / bins_per_block);
        int dst_offset = binid % bins_per_block;
        
        // 获取目标块的共享内存指针
        int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);
        
        // 原子操作更新直方图bin
        atomicAdd(dst_smem + dst_offset, 1);
    }
    // 簇内同步，确保所有分布式共享内存操作完成
    cluster.sync();

    // 使用本地分布式共享内存直方图更新全局内存中的直方图
    int *lbins = bins + cluster.block_rank() * bins_per_block;
    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
    {
        atomicAdd(&lbins[i], smem[i]);
    }
}

// 启动核函数的配置和调用
void launch_histogram_kernel(int *d_bins, const int nbins, const int *d_input, size_t array_size, int threads_per_block)
{
    // 计算网格和块的配置
    int blocks = (array_size + threads_per_block - 1) / threads_per_block;
    
    // 设置CUDA启动配置
    cudaLaunchConfig_t config = {0};
    config.gridDim = blocks;
    config.blockDim = threads_per_block;
    
    int cluster_size = 2;  // 线程簇大小
    int nbins_per_block = nbins / cluster_size;  // 每个块的bin数量
    
    config.dynamicSmemBytes = nbins_per_block * sizeof(int);  // 动态共享内存大小
    cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);
    
    // 设置线程簇属性
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    
    config.numAttrs = 1;
    config.attrs = attribute;
    
    // 启动核函数
    cudaLaunchKernelEx(&config, clusterHist_kernel, d_bins, nbins, nbins_per_block, d_input, array_size);
}

int main()
{
    const int array_size = 1024;  // 输入数据大小
    const int nbins = 64;  // 直方图bin数
    const int threads_per_block = 256;  // 每个块的线程数
    
    int *d_input, *d_bins;

    // 分配device内存
    cudaMalloc(&d_input, array_size * sizeof(int));
    cudaMalloc(&d_bins, nbins * sizeof(int));
    cudaMemset(d_bins, 0, nbins * sizeof(int));  // 初始化直方图bin为0

    // 启动直方图计算核函数
    launch_histogram_kernel(d_bins, nbins, d_input, array_size, threads_per_block);

    // 释放device内存
    cudaFree(d_input);
    cudaFree(d_bins);

    return 0;
}

```

## 2.4  **锁页内存  Page-Locked Memory**

在学习**Page-Locked Memory之前，需要了解页（Page）的概念**

为了更好的管理计算机内存，在与物理器件相对应的内存的物理地址的基础上，对其进行抽象化，定义为虚拟（逻辑）内存。

- **页（Page）**：
    - 逻辑内存被划分成固定大小的块，通常为 4KB 或更大。
    - 每个程序的地址空间由多个页组成。
- **页框（Page Frame）**：
    - 物理内存被划分成与页相同大小的块。
    - 页框是物理内存中的存储单元。
- **页表（Page Table）**：
    - 操作系统维护的一个数据结构，用于记录逻辑页和物理页框之间的映射关系。
    - 每个进程有一个独立的页表。
- **虚拟地址（Virtual Address）**：
    - 程序使用的地址，由操作系统和硬件共同管理。
    - 由页号和页内偏移量组成。
- **物理地址（Physical Address）**：
    - 实际的内存地址。
    - 由页框号和页内偏移量组成。

通过使用逻辑内存这一概念，可以更好的管理实际上的物理内存，并支持使用Swap的操作去缓解内存容量不足的情况。

在这些概念的基础上，有分页这一内存管理概念：分页允许操作系统将不连续的内存地址映射到物理内存中的不同位置，并且能够更高效地管理内存资源。与分页这一概念相对应的即为锁页（Page Locking），有时也称为页面锁定或固定内存，它将特定的内存页固定在物理内存中，防止这些页被交换到磁盘或进行分页操作。锁页的主要目的是确保内存页始终驻留在物理内存中，以提高系统的性能和可靠性，特别是在实时或高性能计算中（其实通俗来说就是不会被移到disk上，避免性能损失）

在实际使用中，对不同的情况使用不同的函数来实现分配锁页内存：

- **分配**：使用`cudaHostAlloc()`函数来分配页面锁定内存。
- **释放**：使用`cudaFreeHost()`函数来释放页面锁定内存。
- **转换**：使用`cudaHostRegister()`函数将通过`malloc()`分配的普通内存转换为页面锁定内存。

同时，存在一些特定的页面锁定内存

- 可移植内存（Portable Memory）
    - 通过设置特定标志（如`cudaHostAllocPortable`或`cudaHostRegisterPortable`），可以使页面锁定内存块在系统中的任何device上使用。
- 写组合内存（Write-Combing Memory）
    - 
    
    使用`cudaHostAllocWriteCombined`标志分配的内存，释放了主机的L1和L2缓存资源，提高了PCI Express总线传输性能。适用于只发生主机写入的内存。
    
- 映射内存（Mapped Memory）
    - 使用`cudaHostAllocMapped`或`cudaHostRegisterMapped`标志分配的内存可以映射到device的地址空间，允许直接从内核访问主机内存。

下面提供了一个简单的使用分页内存的例子

```cpp
#include <cuda_runtime.h>
#include <iostream>

int main() {
    size_t size = 1024 * 1024; // 1MB大小的内存
    float *h_pageLockedData;

    // 分配页面锁定内存
    cudaError_t err = cudaHostAlloc((void**)&h_pageLockedData, size * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating page-locked memory: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    float *d_data;
    // 分配device内存
    err = cudaMalloc((void**)&d_data, size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory: " << cudaGetErrorString(err) << std::endl;
        cudaFreeHost(h_pageLockedData);
        return -1;
    }

    // 页面锁定内存到device内存的数据传输
    err = cudaMemcpy(d_data, h_pageLockedData, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying memory from host to device: " << cudaGetErrorString(err) << std::endl;
    }

    // 清理
    cudaFree(d_data);
    cudaFreeHost(h_pageLockedData);

    return 0;
}

```

## 2.5 纹理内存 **Texture Memory**

纹理内存与表面内存针对纹理与表面设计，在特定情况下拥有更好的性能表现，

有两种不同的 API 可以访问纹理和表面内存：

- 所有设备都支持的纹理引用 API (the texture reference API)，
- 仅在计算能力 3.x 及更高版本的设备上支持的纹理对象 API (The texture object API)。

纹理内存在Kernel中使用设备函数进行进行读取（详细见下文）。使用这些函数读取纹理的过程即为纹理提取（texture fetch），每个纹理提取会指定一个参数，这个参数可以是纹理对象（对于纹理对象API）或纹理引用（对于纹理引用API）。例如，使用纹理对象api实

```cpp
output[y * width + x] = tex2D(texRef, x, y);  // 从纹理中读取数据
```

- Text Object或Text Ref参数：
    - **纹理**：这是被提取的纹理内存部分。纹理可以是线性内存的任何区域或CUDA数组。
        - 纹理对象在运行时创建，创建时指定纹理。
        - 纹理引用在编译时创建，通过运行时函数将纹理引用绑定到纹理来指定纹理。多个不同的纹理引用可以绑定到相同的纹理或内存中重叠的纹理。
    - **维度**：指定纹理的维度，可以是使用一个纹理坐标的一维数组，使用两个纹理坐标的二维数组，或使用三个纹理坐标的三维数组。
        - 数组中的元素称为纹素（texels）。纹理的宽度、高度和深度分别指数组在每个维度上的大小。设备计算能力不同，最大纹理宽度、高度和深度也不同。
    - **纹素类型**：限制为基本的整数和单精度浮点类型，以及由这些基本类型派生的一、二、四分量的向量类型。
    - **读取模式**：可以是 `cudaReadModeNormalizedFloat` 或 `cudaReadModeElementType`。
        - 如果是 `cudaReadModeNormalizedFloat`，并且纹素类型是16位或8位整数类型，纹理提取返回的值会被转换为浮点类型，整数类型的全范围映射为[0.0, 1.0]（无符号整数）或[-1.0, 1.0]（有符号整数）。
        - 例如，无符号8位纹素值0xff读取为1。如果是 `cudaReadModeElementType`，则不进行转换。
    - **坐标归一化标志**：默认情况下，纹理通过浮点坐标引用，坐标范围为[0, N-1]，其中N是对应维度的纹理大小。
        - 例如，大小为64x32的纹理，其x和y维度的坐标范围分别为[0, 63]和[0, 31]。归一化的纹理坐标使得坐标范围为[0.0, 1.0-1/N]，所以相同的64x32纹理在x和y维度的坐标范围为[0, 1-1/N]。归一化纹理坐标对于某些应用来说更加适合，使纹理坐标与纹理大小无关。
    - 寻址方式：设备函数可以使用超出范围的坐标调用。地址模式定义了这种情况下的处理方式。
        - 默认地址模式是将坐标限制在有效范围内：非归一化坐标为[0, N)，归一化坐标为[0.0, 1.0)。
        - 如果指定了边界模式，超出范围的纹理坐标提取返回零。
        - 对于归一化坐标，还有包裹模式和镜像模式可用。
            - 使用包裹模式时，每个坐标x转换为 `frac(x) = x - floor(x)`，其中 `floor(x)` 是不大于x的最大整数。
            - 使用镜像模式时，坐标x转换为 `frac(x)`（如果 `floor(x)` 是偶数）或 `1 - frac(x)`（如果 `floor(x)` 是奇数）。
        - 地址模式指定为大小为三的数组，数组的第一个、第二个和第三个元素分别指定第一个、第二个和第三个纹理坐标的地址模式。地址模式可以是 `cudaAddressModeBorder`、`cudaAddressModeClamp`、`cudaAddressModeWrap` 和 `cudaAddressModeMirror`。 `cudaAddressModeWrap` 和 `cudaAddressModeMirror` 仅支持归一化纹理坐标。
    - **过滤模式**：指定纹理提取时返回值的计算方式。
        - 线性纹理过滤只能用于返回浮点数据的纹理。它在相邻纹素之间进行低精度插值。
        - 当启用时，读取纹理提取位置周围的纹素，并基于纹理坐标在纹素之间的位置进行插值。
        - 一维纹理进行简单线性插值，二维纹理进行双线性插值，三维纹理进行三线性插值。
        - 过滤模式可以是 `cudaFilterModePoint` 或 `cudaFilterModeLinear`。
            - `cudaFilterModePoint`返回是最接近输入纹理坐标的纹素值。
            - `cudaFilterModeLinear`返回是最接近输入纹理坐标的两个（一维纹理）、四个（二维纹理）或八个（三维纹理）纹素的线性插值。`cudaFilterModeLinear` 仅对返回浮点类型的值有效。

### 2.5.1 纹理对象 API Text Object

一个纹理对象可以使用 `cudaCreateTextureObject()` 创建，该函数需要一个类型为 `struct cudaResourceDesc` 的资源描述符（用于指定纹理）和一个纹理描述符。纹理描述符定义如下：

```cpp
struct cudaTextureDesc
{
    enum cudaTextureAddressMode addressMode[3];
    enum cudaTextureFilterMode  filterMode;
    enum cudaTextureReadMode    readMode;
    int                         sRGB;
    int                         normalizedCoords;
    unsigned int                maxAnisotropy;
    enum cudaTextureFilterMode  mipmapFilterMode;
    float                       mipmapLevelBias;
    float                       minMipmapLevelClamp;
    float                       maxMipmapLevelClamp;
};

```

- `addressMode` 指定地址模式；
- `filterMode` 指定过滤模式；
- `readMode` 指定读取模式；
- `normalizedCoords` 指定纹理坐标是否归一化；

```cpp
// 简单的变换内核
__global__ void transformKernel(float* output,cudaTextureObject_t texObj,
int width, int height,float theta) 
{
    // 计算归一化的纹理坐标
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    float u = x / (float)width;
    float v = y / (float)height;

    // 变换坐标
    u -= 0.5f;
    v -= 0.5f; 
    float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
    float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

    // 从纹理读取并写入全局内存
    output[y * width + x] = tex2D<float>(texObj, tu, tv);
}

```

```cpp
    // 指定纹理
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // 指定纹理对象参数
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // 创建纹理对象
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
```

### 2.5.2 16位浮点纹理支持

CUDA C++不直接支持16位浮点数的数据类型，因此需要使用内置函数进行转换

- **`__float2half_rn(float)`**：
    - 该函数将32位浮点数转换为16位浮点数。
    - `rn`表示最近偶数舍入模式（round to nearest even）。
- **`__half2float(unsigned short)`**：
    - 该函数将16位浮点数转换为32位浮点数。
    - 参数类型为无符号短整型（unsigned short），因为CUDA不直接支持half类型。

在纹理提取过程中，16位浮点数会被自动提升为32位浮点数，然后进行纹理过滤操作。这确保了过滤操作具有足够的精度。

### 2.5.3 分层纹理 Plane Texture

层纹理是一种特殊的纹理类型，它由多个相同维度、大小和数据类型的纹理层组成。每一层都是独立的纹理，但它们一起组成一个整体结构。

- **一维层纹理**：每层是一维纹理。
    - 使用 `tex1DLayered(tex, x, layer)` 进行访问，其中 `x` 是浮点坐标，`layer` 是整数索引。
- **二维层纹理**：每层是二维纹理。
    - 使用 `tex2DLayered(tex, x, y, layer)` 进行访问，其中 `x` 和 `y` 是浮点坐标，`layer` 是整数索引。

下面的例子展示了如何创建一个简单的一维层纹理并进行访问：

```cpp
cudaExtent extent = make_cudaExtent(width, 0, layers);
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
cudaArray_t cuArray;
cudaMalloc3DArray(&cuArray, &channelDesc, extent, cudaArrayLayered);

__global__ void kernel(cudaTextureObject_t tex, float* output, int width, int height, int layers) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int layer = 0; layer < layers; ++layer) {
            float u = x / (float)width;
            float v = y / (float)height;
            output[(layer * height + y) * width + x] = tex2DLayered<float>(tex, u, v, layer);
        }
    }
}
```

纹理过滤仅在单个层内进行，不跨层进行。这意味着每个层是独立的，过滤操作不会影响其他层的数据

### 2.5.4 **立方体纹理 Cubemap Texture**

立方体纹理是一种特殊类型的纹理，它有六个面，每个面都是相同尺寸的正方形纹理。这六个面一起组成一个立方体，并通过三个坐标（x, y, z）进行寻址。这个立方体贴图常用于环境映射和反射计算。

立方体贴图纹理只能通过 `cudaMalloc3DArray()` 函数创建，并且需要使用 `cudaArrayCubemap` 标志。例如：

```cpp
cudaExtent extent = make_cudaExtent(width, height, 6);
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
cudaArray_t cuArray;
cudaMalloc3DArray(&cuArray, &channelDesc, extent, cudaArrayCubemap);
```

立方体贴图纹理的访问

立方体贴图纹理的访问通过 `texCubemap()` 设备函数实现，使用三个坐标（x, y, z）来确定采样点。这些坐标表示一个从立方体中心发出的方向向量，指向立方体的一个面及其内的一个纹素。

具体来说，最大幅度的坐标确定了对应的面，然后在该面内使用剩余的两个坐标进行纹素的采样。

| 条件 | 面 | m | s | t |
| --- | --- | --- | --- | --- |
| |x| > |y| and |x| > |z| | x ≥ 0 | 0 | x | -z |
| |x| > |y| and |x| > |z| | x < 0 | 1 | -x | z |
| |y| > |x| and |y| > |z| | y ≥ 0 | 2 | y | x |
| |y| > |x| and |y| > |z| | y < 0 | 3 | -y | x |
| |z| > |x| and |z| > |y| | z ≥ 0 | 4 | z | x |
| |z| > |x| and |z| > |y| | z < 0 | 5 | -z | -x |

### 2.5.5 **分层立方体纹理 Cubemap Layered Textures**

立方图层纹理是一种层状纹理，其层是具有相同维度的立方体贴图。

立方图层纹理使用一个整数索引和三个浮点纹理坐标进行寻址；索引表示序列中的一个立方体贴图，而坐标则指向该立方体贴图中的一个纹素。

立方图层纹理只能通过调用`cudaMalloc3DArray()`并使用`cudaArrayLayered`

和`cudaArrayCubemap`标志来分配 CUDA 数组。立方图层纹理通过设备函数`texCubemapLayered()`进行获取。纹理过滤（请参见纹理获取）仅在层内进行，而不会跨层进行。

### 2.5.6 **纹理聚集 Texture Gather**

纹理聚集是一种仅适用于二维纹理的特殊的纹理获取操作。它由 `tex2Dgather()` 函数执行，该函数的参数与 `tex2D()` 相同，外加一个额外的 `comp` 参数，该参数可以为 0、1、2 或 3（参见 `tex2Dgather()`）。`tex2Dgather()` 返回四个 32 位数值，这些数值对应于在常规纹理获取期间用于双线性过滤的四个纹素的 `comp` 组件的值。

例如，如果这些纹素的值为 (253, 20, 31, 255)、(250, 25, 29, 254)、(249, 16, 37, 253)、(251, 22, 30, 250)，而 `comp` 为 2，`tex2Dgather()` 将返回 (31, 29, 37, 30)。

请注意，纹理坐标的计算仅具有 8 位的小数精度。因此，`tex2Dgather()` 可能会在 `tex2D()` 使用 1.0 作为其权重之一（α 或 β）时返回意外结果（参见线性过滤）。例如，当 x 纹理坐标为 2.49805 时：xB = x - 0.5 = 1.99805，但 xB 的小数部分以 8 位定点格式存储。由于 0.99805 更接近 256/256 而不是 255/256，xB 的值为 2。在这种情况下，`tex2Dgather()` 将返回 x 中的索引 2 和 3，而不是索引 1 和 2。

纹理聚集仅支持通过 `cudaArrayTextureGather` 标志创建的 CUDA 数组，且宽度和高度必须小于表 15 中指定的纹理聚集最大值，这比常规纹理获取的最大值要小。纹理聚集仅在计算能力为 2.0 及以上的设备上支持。

## 2.6 表面内存 Surface Memory

对于计算能力为 2.0 及以上的设备，可以通过 `cudaArraySurfaceLoadStore` 标志创建的 CUDA 数组（参见立方图表面部分）来进行读写操作。这些操作可以通过表面Object或表面Ref，使用相关的表面函数来实现。

### 2.6.1 Surface Object API

Surface Object是通过 `cudaCreateSurfaceObject()` 函数，从类型为 `struct cudaResourceDesc` 的资源描述符中创建的。以下代码示例展示了如何对纹理应用一个简单的转换核函数。

```cpp
__global__ void copyKernel(cudaSurfaceObject_t inputSurfObj,
                           cudaSurfaceObject_t outputSurfObj,
                           int width, int height) 
{
    // 计算表面坐标
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        uchar4 data; // 从输入表面读取
        surf2Dread(&data, inputSurfObj, x * 4, y);
        // 写入输出表面
        surf2Dwrite(data, outputSurfObj, x * 4, y);
    }
}

// 主机代码
int main()
{
    const int height = 1024;
    const int width = 1024;
    // 分配并设置一些主机数据
    unsigned char *h_data = (unsigned char *)std::malloc(sizeof(unsigned char) * width * height * 4);
    for (int i = 0; i < height * width * 4; ++i)
        h_data[i] = i;
    
    // 在设备内存中分配 CUDA 数组
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuInputArray;
    cudaMallocArray(&cuInputArray, &channelDesc, width, height, cudaArraySurfaceLoadStore);
    cudaArray_t cuOutputArray;
    cudaMallocArray(&cuOutputArray, &channelDesc, width, height, cudaArraySurfaceLoadStore);
    
    // 设置源的行距（内存中指向 src 的 2D 数组的宽度，以字节为单位，包括填充），我们没有填充
    const size_t spitch = 4 * width * sizeof(unsigned char);
    // 将位于主机内存中 h_data 地址的数据复制到设备内存
    cudaMemcpy2DToArray(cuInputArray, 0, 0, h_data, spitch, 4 * width * sizeof(unsigned char), height, cudaMemcpyHostToDevice);
    
    // 指定表面
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    
    // 创建表面对象
    resDesc.res.array.array = cuInputArray;
    cudaSurfaceObject_t inputSurfObj = 0;
    cudaCreateSurfaceObject(&inputSurfObj, &resDesc);
    resDesc.res.array.array = cuOutputArray;
    cudaSurfaceObject_t outputSurfObj = 0;
    cudaCreateSurfaceObject(&outputSurfObj, &resDesc);
    
    // 调用核函数
    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x, (height + threadsperBlock.y - 1) / threadsperBlock.y);
    copyKernel<<<numBlocks, threadsperBlock>>>(inputSurfObj, outputSurfObj, width, height);
    
    // 将数据从设备复制回主机
    cudaMemcpy2DFromArray(h_data, spitch, cuOutputArray, 0, 0, 4 * width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);
    
    // 销毁表面对象
    cudaDestroySurfaceObject(inputSurfObj);
    cudaDestroySurfaceObject(outputSurfObj);
    
    // 释放设备内存
    cudaFreeArray(cuInputArray);
    cudaFreeArray(cuOutputArray);
    
    // 释放主机内存
    free(h_data);
    
    return 0;
}

```

### 2.6.2  Cubemap Surfaces

立方图表面通过 `surfCubemapread()` 和 `surfCubemapwrite()` 函数进行访问，作为一个二维层状表面，即使用一个整数索引表示一个面，并使用两个浮点纹理坐标来定位该面对应层中的一个纹素。

### 2.6.3 Cubemap Layered Surfaces

立方图层状表面通过 `surfCubemapLayeredread()` 和 `surfCubemapLayeredwrite()` 函数进行访问，作为一个二维层状表面，即使用一个整数索引表示其中一个立方图的一个面，并使用两个浮点纹理坐标来定位该面对应层中的一个纹素。各个面的顺序如表2所示，因此，例如，索引 ((2 * 6) + 3) 访问第三个立方图的第四个面。

## 2.7 CUDA 数组

CUDA 数组（CUDA Array）是一种不透明的内存布局，经过优化以进行纹理获取。它们可以是一维、二维或三维的，由元素组成，每个元素可以具有1、2或4个组件，这些组件可以是有符号或无符号的8位、16位或32位整数，16位浮点数或32位浮点数。CUDA  数组只能通过纹理获取（如纹理内存中描述）或表面读写（如表面内存中描述）由内核访问。

### 2.7.1  读/写一致性

纹理和表面内存（Texture and Surface Memory）在运行时将会使用Cache以进行加速，然而，对同一个Kernel调用中，这些缓存不会与全局内存（Global Memory）和表面内存的（Surface Memory）内容保持一致。在此条件下，如果在一个Kernel函数中对某个内存地址进行了全局内存的写入或表面内存的写入，并随后对该地址进行纹理/表面获取，这将返回一个未定义的数据。换句话说，一个thread只能安全地读取某个纹理或表面内存位置，前提是该内存位置已经被之前的内核调用或内存复制更新，而不是被同一内核调用中的同一thread或另一个thread更新。

这里给出一个错误操作示例：

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void exampleKernel(cudaTextureObject_t texObj, float* globalMem, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;

        // 写入全局内存
        globalMem[idx] = x * 0.1f + y * 0.1f;

        // 尝试读取纹理数据
        float texValue = tex2D<float>(texObj, x, y);

        // 输出数据
        printf("Thread (%d, %d): Global Memory = %f, Texture Memory = %f\n", x, y, globalMem[idx], texValue);
    }
}

int main()
{
    const int width = 8;
    const int height = 8;
    const int size = width * height * sizeof(float);

    // 分配主机和设备内存
    float h_data[width * height];
    float* d_globalMem;
    cudaMalloc((void**)&d_globalMem, size);

    // 填充主机数据
    for (int i = 0; i < width * height; i++) {
        h_data[i] = static_cast<float>(i);
    }

    // 创建CUDA数组并复制数据
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    cudaMemcpyToArray(cuArray, 0, 0, h_data, size, cudaMemcpyHostToDevice);

    // 创建纹理对象
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    // 调用内核
    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    exampleKernel<<<numBlocks, threadsPerBlock>>>(texObj, d_globalMem, width, height);

    // 同步设备
    cudaDeviceSynchronize();

    // 清理资源
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_globalMem);

    return 0;
}

```

## 2.8 模式切换

具有显示输出的 GPU 会将一些 DRAM 内存专用于所谓的主表面（primary surface），该表面用于刷新用户所查看的显示设备。当用户通过更改显示分辨率或色深（使用 NVIDIA 控制面板或 Windows 的显示控制面板）来发起显示模式切换时，主表面所需的内存量会发生变化。例如，如果用户将显示分辨率从 1280x1024x32 位更改为 1600x1200x32 位，系统必须为主表面分配 7.68 MB 而不是 5.24 MB。（启用抗锯齿的全屏图形应用程序可能需要更多的显示内存用于主表面。）在 Windows 上，其他可能引发显示模式切换的事件包括启动全屏 DirectX 应用程序，按下 Alt+Tab 以任务切换离开全屏 DirectX 应用程序，或按下 Ctrl+Alt+Del 锁定计算机。

如果模式切换增加了主表面所需的内存量，系统可能不得不侵占专用于 CUDA 应用程序的内存分配。因此，模式切换会导致对 CUDA 运行时的任何调用失败并返回无效上下文错误。

# 3. 异步并发

异步并发是cuda编程中的重要概念，其定义了六种可以并发执行的操作：

- host 上的计算；
- device 上的计算；
- 从host到device的内存传输；
- 从device到host的内存传输；
- 在同一device内的内存传输；
- device之间的内存传输。

具体而言，是否可以进行并发执行取决于Device的具体型号，者通过一些标识符进行标志，如下所示：

- Host 和 Device 之间的并发执行
- Kernels之间的并发执行
    - 标志：`concurrentKernels`=1
- 数据传输和Kernel的并发执行
    - 标志：`asyncEngineCount` >0
- Devices之间的内存传输
    - 标志：`asyncEngineCount`=2

## 3.1  Stream

应用程序通过Stream来管理上述并发操作。Stream是按顺序执行的命令序列（这些命令可能由不同的 host 线程发出）。另一方面，不同的Stream之间可以乱序或并发地执行命令；

在CUDA编程中，两个 Stream 之间的操作可以重叠执行，从而提高程序的执行效率。然而，Stream之间的重叠运行程度取决于每个 Stream 中命令的顺序，以及device是否支持对应的功能（详见上文）

这里给出一份实现了数据传输与内核执行重叠、同时内核执行、并发数据传输这三种并发操作的代码：

- **数据传输与内核执行重叠**：
    - 使用`cudaMemcpyAsync`进行异步数据传输，而不是`cudaMemcpy`。这允许数据传输操作和内核执行在不同的Stream中同时进行。
    - 例如，在`streams[0]`中，数据传输从主机到device（`cudaMemcpyAsync`）和内核执行（`MyKernel`）可以重叠运行，因为它们在同一个Stream中异步执行。
- **同时内核执行**：
    - 内核函数`MyKernel`在多个Stream中并发执行。例如，`streams[0]`、`streams[1]`和`streams[2]`中的内核可以同时在GPU上运行。
- **并发数据传输**：
    - 在多个Stream中并行进行数据传输操作。例如，`streams[0]`、`streams[1]`和`streams[2]`中的数据传输操作（`cudaMemcpyAsync`）可以同时进行。

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void MyKernel(float* data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] += 1.0f;
    }
}

int main() {
    const int size = 1024 * 1024; // 1MB 数据大小
    const int numStreams = 3;
    float* h_data[numStreams];
    float* d_data[numStreams];

    // 分配锁页主机内存和device内存
    for (int i = 0; i < numStreams; ++i) {
        cudaMallocHost(&h_data[i], size * sizeof(float)); // 锁页主机内存
        cudaMalloc(&d_data[i], size * sizeof(float)); // device内存
        // 初始化主机数据
        for (int j = 0; j < size; ++j) {
            h_data[i][j] = static_cast<float>(j);
        }
    }

    // 创建多个Stream
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // 启动异步内存复制和内核执行
    for (int i = 0; i < numStreams; ++i) {
        // 异步内存复制：从主机到device
        cudaMemcpyAsync(d_data[i], h_data[i], size * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        // 启动内核
        MyKernel<<<(size + 255) / 256, 256, 0, streams[i]>>>(d_data[i], size);
        // 异步内存复制：从device到主机
        cudaMemcpyAsync(h_data[i], d_data[i], size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    // 同步Stream
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    // 验证结果
    bool success = true;
    for (int i = 0; i < numStreams; ++i) {
        for (int j = 0; j < size; ++j) {
            if (h_data[i][j] != static_cast<float>(j) + 1.0f) {
                success = false;
                break;
            }
        }
    }

    if (success) {
        std::cout << "success test!" << std::endl;
    } else {
        std::cout << "fail test!" << std::endl;
    }

    // 清理资源
    for (int i = 0; i < numStreams; ++i) {
        cudaFreeHost(h_data[i]);
        cudaFree(d_data[i]);
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}
```

## 3.2 **Graphs**

CUDA Graphs 是 NVIDIA CUDA 提供的一种新特性，用于捕获和重用 GPU 工作负载的执行模式。通过使用 CUDA Graphs，可以显著减少 CPU 与 GPU 之间的通信开销，提高 GPU 程序的性能和效率

为何可以实现优化：当你将一个内核（Kernel）放入一个Stream中时，主机驱动程序需要执行一系列准备工作，这些工作是为了在GPU上正确执行内核所必需的。这些设置和启动内核的操作会产生一定的开销，对于一些较小的kernel函数，准备工作所需的时间相对于内核的实际执行时间而言，会显得特别长，从而表现为性能损失

使用 Graphs 的工作提交分为三个阶段：定义、实例化和执行。

- 定义
    - 在定义阶段，程序会创建Graph，并描述其中的操作和它们之间的依赖关系，列出所有任务以及它们的先后顺序。
- 实例化
    - 获取这个Graph的快照，并对其进行验证
    - 执行大部分的设置和初始化工作，以减少启动时需要的工作量
    - 生成的实例称为“可执行图”（Executable Graph）
- 运行
    - 可执行图（Executable Graph）可以在Stream中启动

### 3.2.1 基本概念

1. **Graph（图）**：
    - 图是由一系列操作（如内核执行、内存拷贝等）和这些操作之间的依赖关系组成的有向无环图（DAG）。
    - 图可以包含多个节点，每个节点代表一个操作。
2. **Node（节点）**：
    - 节点是图中的基本单元，代表具体的操作。
    - 常见的节点类型包括内核节点、内存拷贝节点、内存设置节点等。
3. **Graph Capture（图捕获）**：
    - 图捕获是指将一系列操作捕获到一个图中。可以通过调用相应的 API 来捕获这些操作，并构建图。
4. **Graph Execution（图执行）**：
    - 一旦图构建完成，可以多次执行该图。执行图时，CUDA 运行时会根据图中的依赖关系调度各个操作。

节点类型：

- Kernel 函数
- CPU函数调用
- 内存拷贝
- 内存设置
- 空节点
- 等待事件
- 记录事件
- 发出外部信号量的信号
- 等待外部信号量
- 子图：执行单独的嵌套图。

### 3.2.2 构建Graph

Graph的构建主要通过两种方式：

- 显式API
    
    ```cpp
    #include <cuda_runtime.h>
    #include <iostream>
    
    __global__ void MyKernel(float* data, int size) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < size) {
            data[idx] += 1.0f;
        }
    }
    
    int main() {
        const int size = 1024; // 数据大小
        float* h_data;
        float* d_data;
    
        // 分配锁页主机内存和device内存
        cudaMallocHost(&h_data, size * sizeof(float)); // 锁页主机内存
        cudaMalloc(&d_data, size * sizeof(float)); // device内存
    
        // 初始化主机数据
        for (int i = 0; i < size; ++i) {
            h_data[i] = static_cast<float>(i);
        }
    
        // 创建一个空图
        cudaGraph_t graph;
        cudaGraphCreate(&graph, 0);
    
        // 创建一个内存拷贝节点（从主机到device）
        cudaMemcpy3DParms copyParams = {nullptr};
        copyParams.srcPtr = make_cudaPitchedPtr(h_data, size * sizeof(float), size, 1);
        copyParams.dstPtr = make_cudaPitchedPtr(d_data, size * sizeof(float), size, 1);
        copyParams.extent = make_cudaExtent(size * sizeof(float), 1, 1);
        copyParams.kind = cudaMemcpyHostToDevice;
    
        cudaGraphNode_t memcpyNode;
        cudaGraphAddMemcpyNode(&memcpyNode, graph, nullptr, 0, &copyParams);
    
        // 创建一个内核节点
        void* kernelArgs[] = {&d_data, (void *) &size};
        cudaKernelNodeParams kernelNodeParams = {};
        kernelNodeParams.func = (void*)MyKernel;
        kernelNodeParams.gridDim = dim3((size + 255) / 256);
        kernelNodeParams.blockDim = dim3(256);
        kernelNodeParams.kernelParams = kernelArgs;
        kernelNodeParams.extra = nullptr;
    
        cudaGraphNode_t kernelNode;
        cudaGraphAddKernelNode(&kernelNode, graph, &memcpyNode, 1, &kernelNodeParams);
    
        // 创建图实例
        cudaGraphExec_t graphExec;
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    
        // 执行图实例
        cudaGraphLaunch(graphExec, 0);
        cudaDeviceSynchronize();
    
        // 验证结果
        cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
        bool success = true;
        for (int i = 0; i < size; ++i) {
            if (h_data[i] != static_cast<float>(i) + 1.0f) {
                success = false;
                break;
            }
        }
    
        if (success) {
            std::cout << "Success!" << std::endl;
        } else {
            std::cout << "Fail!" << std::endl;
        }
    
        // 销毁图实例和图
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
    
        // 清理资源
        cudaFreeHost(h_data);
        cudaFree(d_data);
    
        return 0;
    }
    
    ```
    
    - **创建图**：
        - 使用 `cudaGraphCreate` 创建一个空图 `graph`。
    - **添加内存拷贝节点**：
        - 使用 `cudaMemcpy3DParms` 结构体设置内存拷贝参数。
        - 使用 `cudaGraphAddMemcpyNode` 函数将内存拷贝节点添加到图中。
    - **添加内核节点**：
        - 使用 `cudaKernelNodeParams` 结构体设置内核参数。
        - 使用 `cudaGraphAddKernelNode` 函数将内核节点添加到图中。
    - **创建图实例**：
        - 使用 `cudaGraphInstantiate` 从图 `graph` 创建一个可执行的图实例 `graphExec`。
    - **执行图实例**：
        - 使用 `cudaGraphLaunch` 函数执行图实例。
    - **验证结果**：
        - 检查主机内存中的数据是否符合预期。
    - **销毁图实例和图**：
        - 使用 `cudaGraphExecDestroy` 和 `cudaGraphDestroy` 函数销毁图实例和图。
    - **清理资源**：
        - 释放分配的主机内存和device内存。
- Stream流捕获
    
    ```cpp
    #include <cuda_runtime.h>
    #include <iostream>
    
    __global__ void MyKernel(float* data, int size) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < size) {
            data[idx] += 1.0f;
        }
    }
    
    int main() {
        const int size = 1024; // 数据大小
        float* h_data;
        float* d_data;
    
        // 分配锁页主机内存和device内存
        cudaMallocHost(&h_data, size * sizeof(float)); // 锁页主机内存
        cudaMalloc(&d_data, size * sizeof(float)); // device内存
    
        // 初始化主机数据
        for (int i = 0; i < size; ++i) {
            h_data[i] = static_cast<float>(i);
        }
    
        // 创建一个Stream
        cudaStream_t stream;
        cudaStreamCreate(&stream);
    
        // 启动Stream捕获
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
        // 在Stream中执行操作
        cudaMemcpyAsync(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice, stream);
        MyKernel<<<(size + 255) / 256, 256, 0, stream>>>(d_data, size);
        cudaMemcpyAsync(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    
        // 结束Stream捕获并生成图
        cudaGraph_t graph;
        cudaStreamEndCapture(stream, &graph);
    
        // 创建图实例
        cudaGraphExec_t graphExec;
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    
        // 执行图实例
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
    
        // 验证结果
        bool success = true;
        for (int i = 0; i < size; ++i) {
            if (h_data[i] != static_cast<float>(i) + 1.0f) {
                success = false;
                break;
            }
        }
    
        if (success) {
            std::cout << "Succeed" << std::endl;
        } else {
            std::cout << "Failed!" << std::endl;
        }
    
        // 销毁图实例和图
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
    
        // 清理资源
        cudaFreeHost(h_data);
        cudaFree(d_data);
        cudaStreamDestroy(stream);
    
        return 0;
    }
    
    ```
    
    - **创建Stream**：
        - 使用 `cudaStreamCreate` 创建一个Stream `stream`。
    - **启动Stream捕获**：
        - 使用 `cudaStreamBeginCapture` 启动Stream捕获模式。这里使用的是全局捕获模式 `cudaStreamCaptureModeGlobal`。
    - **在Stream中执行操作**：
        - 在捕获模式下，将一系列操作放入Stream中，包括内存拷贝和内核执行。
    - **结束Stream捕获并生成图**：
        - 使用 `cudaStreamEndCapture` 结束Stream捕获，并生成一个CUDA Graph `graph`。
    - **创建图实例**：
        - 使用 `cudaGraphInstantiate` 从图 `graph` 创建一个可执行的图实例 `graphExec`。
    - **执行图实例**：
        - 使用 `cudaGraphLaunch` 函数在指定的Stream中执行图实例。
    - **验证结果**：
        - 检查主机内存中的数据是否符合预期。
    - **销毁图实例和图**：
        - 使用 `cudaGraphExecDestroy` 和 `cudaGraphDestroy` 函数销毁图实例和图。
    - **清理资源**：
        - 释放分配的主机内存和device内存，销毁Stream。

可以看出，显式构建Graph可以更精确、可控的描述Graph，但是代码也较为复杂。

值得注意的是，Stream Capture功能可以用于任何CUDA Stream，但不包括`cudaStreamLegacy`，关于此Stream的描述请参照第一张对Stream的描述。

mark：这里牵涉到了Event相关的内容，待会再写

### 3.2.3 更新Graph

图的更新机制主要分为两种：

- **全图更新**：
    - 适用于大量节点更新或调用者不清楚图的拓扑结构（如通过Stream Capture生成的图）。
    - 用户提供一个拓扑相同的`cudaGraph_t`对象，其中节点包含更新后的参数。
- **单节点更新**：
    - 适用于少量节点更新且用户拥有需要更新节点的句柄的情况。
    - 只更新指定节点的参数，跳过未修改节点的拓扑检查与比较，通常更高效。

下面分别给出两种更新机制的简单实现

- 单节点更新
    
    ```cpp
    #include <cuda_runtime.h>
    #include <iostream>
    
    // 简单的CUDA内核
    __global__ void MyKernel(float* data, int size) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < size) {
            data[idx] += 1.0f;
        }
    }
    
    int main() {
        const int size = 1024;
        float* h_data;
        float* d_data;
    
        // 分配锁页主机内存和device内存
        cudaMallocHost(&h_data, size * sizeof(float)); // 锁页主机内存
        cudaMalloc(&d_data, size * sizeof(float)); // device内存
    
        // 初始化主机数据
        for (int i = 0; i < size; ++i) {
            h_data[i] = static_cast<float>(i);
        }
    
        // 创建一个空图
        cudaGraph_t graph;
        cudaGraphCreate(&graph, 0);
    
        // 创建一个内存拷贝节点（从主机到device）
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr = make_cudaPitchedPtr(h_data, size * sizeof(float), size, 1);
        copyParams.dstPtr = make_cudaPitchedPtr(d_data, size * sizeof(float), size, 1);
        copyParams.extent = make_cudaExtent(size * sizeof(float), 1, 1);
        copyParams.kind = cudaMemcpyHostToDevice;
    
        cudaGraphNode_t memcpyNode;
        cudaGraphAddMemcpyNode(&memcpyNode, graph, nullptr, 0, &copyParams);
    
        // 创建一个内核节点
        void* kernelArgs[] = {&d_data, (void *) &size};
        cudaKernelNodeParams kernelNodeParams = {};
        kernelNodeParams.func = (void*)MyKernel;
        kernelNodeParams.gridDim = dim3((size + 255) / 256);
        kernelNodeParams.blockDim = dim3(256);
        kernelNodeParams.kernelParams = kernelArgs;
        kernelNodeParams.extra = nullptr;
    
        cudaGraphNode_t kernelNode;
        cudaGraphAddKernelNode(&kernelNode, graph, &memcpyNode, 1, &kernelNodeParams);
    
        // 创建图实例
        cudaGraphExec_t graphExec;
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    
        // 执行图实例
        cudaGraphLaunch(graphExec, nullptr);
        cudaDeviceSynchronize();
    
        // 更新内核节点的参数（示例）
        int newSize = 2048;
        void* newKernelArgs[] = { &d_data, &newSize };
        cudaKernelNodeParams newKernelNodeParams = kernelNodeParams;
        newKernelNodeParams.kernelParams = newKernelArgs;
        cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &newKernelNodeParams);
    
        // 再次执行图实例
        cudaGraphLaunch(graphExec, nullptr);
        cudaDeviceSynchronize();
    
        // 验证结果
        cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
        bool success = true;
        for (int i = 0; i < size; ++i) {
            if (h_data[i] != static_cast<float>(i) + 2.0f) { // Should be +2.0f because kernel ran twice
                success = false;
                break;
            }
        }
    
        if (success) {
            std::cout << "Succeed!" << std::endl;
        } else {
            std::cout << "Failed!" << std::endl;
        }
    
        // 销毁图实例和图
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
    
        // 清理资源
        cudaFreeHost(h_data);
        cudaFree(d_data);
    
        return 0;
    }
    
    ```
    
- 全图更新
    
    ```cpp
    #include <cuda_runtime.h>
    #include <iostream>
    
    __global__ void MyKernel(float* data, int size) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < size) {
            data[idx] += 1.0f;
        }
    }
    
    void do_cuda_work(cudaStream_t stream) {
        const int size = 1024;
        float* d_data;
    
        // 分配device内存
        cudaMalloc(&d_data, size * sizeof(float));
    
        // 在stream中执行操作
        MyKernel<<<(size + 255) / 256, 256, 0, stream>>>(d_data, size);
        cudaFree(d_data);
    }
    
    int main() {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
    
        cudaGraphExec_t graphExec = nullptr;
    
        for (int i = 0; i < 10; i++) {
            cudaGraph_t graph;
            cudaGraphExecUpdateResult updateResult;
            cudaGraphNode_t errorNode;
    
            // 使用Stream Capture创建图
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            do_cuda_work(stream);
            cudaStreamEndCapture(stream, &graph);
    
            // 尝试更新已实例化的图
            if (graphExec != nullptr) {
                cudaGraphExecUpdate(graphExec, graph, &errorNode, &updateResult);
            }
    
            // 如果第一次迭代或更新失败，则重新实例化图
            if (graphExec == nullptr || updateResult != cudaGraphExecUpdateSuccess) {
                if (graphExec != nullptr) {
                    cudaGraphExecDestroy(graphExec);
                }
                cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
            }
    
            // 销毁图并执行实例化图
            cudaGraphDestroy(graph);
            cudaGraphLaunch(graphExec, stream);
            cudaStreamSynchronize(stream);
        }
    
        // 清理资源
        cudaGraphExecDestroy(graphExec);
        cudaStreamDestroy(stream);
    
        return 0;
    }
    
    ```
    

图更新同样存在一定的限制：

1. 内核节点
    1. 内核节点所属的CUDA Context 必须保持不变。
    2. 不能将最初未使用CUDA动态并行性功能的内核节点更新为使用动态并行性功能的节点。动态并行性允许内核在GPU上启动其他内核，这是一种复杂的功能，要求在图最初定义时明确指定。
2. cudaMemset 和 cudaMemcpy 节点
    1. 分配/映射操作数所指向的CUDAdevice必须保持不变。也就是说，内存操作必须在同一个device上执行，不能跨device进行更新。
    2. 源和目标内存必须从与原始源和目标内存相同的上下文中分配。这确保了内存操作在相同的环境下进行，避免跨上下文操作带来的问题。
    3. 只能更改一维的cudaMemset和cudaMemcpy节点。多维操作（如二维或三维的内存设置和复制）不能通过图更新机制进行更改。
3. 额外的 memcpy 节点限制
    1. 不能修改源或目标位置的内存类型（如`cudaPitchedPtr`、`cudaArray_t`等）或传输类型（如`cudaMemcpyKind`）。这意味着一旦定义了内存操作的类型和方向，这些属性就不能在更新时更改。
4. 外部信号量等待节点和记录节点
    1. 不能修改信号量的数量。信号量用于在不同的CUDA Stream或进程之间同步操作，信号量的数量必须在图定义时固定。
5.  外部信号量等待节点和记录节点
    1. 对主机节点、事件记录节点或事件等待节点的更新没有任何限制。这些节点的更新是灵活的，可以根据需要进行调整。

## 3.3 Event

在CUDA编程中，事件（Events）提供了一种监视device进度和进行准确计时的方法。通过在程序中任意点记录事件，并查询这些事件何时完成，开发者可以有效地监控device上的任务执行进度。

### 3.3.1 创建与销毁事件

```cpp
// 开始时创建事件
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// 使用完毕后销毁事件
cudaEventDestroy(start);
cudaEventDestroy(stop);

```

### 3.3.2 基于Event实现计时

```cpp
cudaEventRecord(start, 0);  // 记录起始事件

for (int i = 0; i < 2; ++i) {
    // 异步地将数据从主机传输到device
    cudaMemcpyAsync(inputDev + i * size, inputHost + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
    // 在device上执行核函数
    MyKernel<<<100, 512, 0, stream[i]>>>(outputDev + i * size, inputDev + i * size, size);
    // 异步地将数据从device传输回主机
    cudaMemcpyAsync(outputHost + i * size, outputDev + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}

cudaEventRecord(stop, 0);  // 记录结束事件
cudaEventSynchronize(stop);  // 同步等待事件完成

float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);  // 计算从start到stop的时间
```

### 3.3.3 Event的同步

当记录一个事件（例如使用 `cudaEventRecord`）并调用 `cudaEventSynchronize` 时，主机线程会等待这个事件完成，即事件之前的所有任务都完成后，主机线程才会继续执行。这是通过阻塞主机线程直到device完成所有相关任务来实现的。例如：

```cpp
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);  // 等待事件完成
```

在这段代码中，主机线程会阻塞在 `cudaEventSynchronize(stop)` 处，直到事件 `stop` 之前的所有任务都完成。

## 3.4 等待策略

对于异步并发操作中，常出现`让步（yield）`、`阻塞（blocking）`和`自旋（spin）`等概念，在cuda编程中，其主要描述了device与host进程之间的关系，详细如下。

### 自旋（Spin）

自旋等待意味着主机线程会在一个忙等待循环中不停地检查device任务是否完成。在这种模式下，线程不会让出CPU时间片，而是不断轮询任务的状态。

- **优点**：延迟最小，因为线程持续占用CPU并快速检测任务完成状态。
- **缺点**：会占用大量CPU资源，因为线程始终处于活跃状态，适合短时间的等待。

在CUDA中，可以通过设置 `cudaDeviceScheduleSpin` 来启用自旋等待：

```cpp
cudaSetDeviceFlags(cudaDeviceScheduleSpin);
```

### 让步（Yield）

让步等待意味着主机线程会在检查device任务状态时主动让出CPU时间片，使其他线程或进程有机会执行。线程会在后续的调度周期中再次检查任务状态。

- **优点**：减少了CPU资源的占用，因为线程让出CPU时间片后其他线程可以执行。
- **缺点**：等待时间可能会略长于自旋等待，因为线程需要等待再次被调度。

在CUDA中，可以通过设置 `cudaDeviceScheduleYield` 来启用让步等待：

```cpp
cudaSetDeviceFlags(cudaDeviceScheduleYield);
```

### 阻塞（Blocking）

阻塞等待意味着主机线程会完全暂停执行，直到device任务完成。线程会进入休眠状态，直到被唤醒以继续执行任务。

- **优点**：最节省CPU资源，因为线程在等待期间不会消耗CPU。
- **缺点**：延迟较高，因为唤醒和调度线程需要一定的时间开销，适合长时间的等待。

在CUDA中，可以通过设置 `cudaDeviceScheduleBlockingSync` 来启用阻塞等待：

```cpp
cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
```

## 3.5 多 Device 管理

### 3.5.1 device 管理

一个host可以同时拥有多个device，并通过简单的语句进行查询与管理：

```cpp
// CUDAdevice的数量
int deviceCount;
// 获取系统中CUDAdevice的数量，并将结果存储在deviceCount中
cudaGetDeviceCount(&deviceCount);

// 当前device的索引
int device;
for (device = 0; device < deviceCount; ++device) {
    // 结构体变量 deviceProp 用于存储device属性
    cudaDeviceProp deviceProp;
    // 获取当前device的属性，并将结果存储在deviceProp中
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n",
           device, deviceProp.major, deviceProp.minor);
}

```

### 3.5.2 选择host线程所操作的device

通过 `cudaSetDevice()` 函数，host 线程可以随时设置当前操作的device。所有的device内存分配和核函数启动都是在当前设置的device上进行的。此外，流（streams）和事件（events）也与当前设置的device相关联。如果没有调用 `cudaSetDevice()`，则默认当前device为device0。

```cpp
size_t size = 1024 * sizeof(float);  
// 将当前device设置为device0
cudaSetDevice(0);                    
// 在device0上分配内存
float* p0;
cudaMalloc(&p0, size);             
// 在device0上启动核函数
MyKernel<<<1000, 128>>>(p0);        
// 将当前device设置为device1
cudaSetDevice(1); 
// 在device1上分配内存
float* p1;
cudaMalloc(&p1, size);               
// 在device1上启动核函数
MyKernel<<<1000, 128>>>(p1);  
```

### 3.5.3 多device下的Stream and Event Behavior

1. Kernel函数的启动必须在与当前device相关联的流中进行
2. 内存复制不受流所关联的device限制
3. `cudaEventRecord()`需要事件与流所关联的Device相同
4. `cudaEventElapsedTime()` 需要两个事件所关联的Device一致
5. `cudaEventSynchronize()` 和 `cudaEventQuery()` 不受关联Device的限制
6. `cudaStreamWaitEvent()` 不受流与关联Device的影响，可用于多个device之间的同步。
7. 对不同Device中的默认流，Device的默认流上的命令可能会相对于其他Device的默认流上的命令无序执行或并发执行

```cpp
cudaSetDevice(0);              
cudaStream_t s0;
cudaStreamCreate(&s0);         
// 在device0的流s0上启动核函数
MyKernel<<<100, 64, 0, s0>>>(); 

cudaSetDevice(1);
cudaStream_t s1;
cudaStreamCreate(&s1); 
// 在device1的流s1上启动核函数
MyKernel<<<100, 64, 0, s1>>>(); 

// 尝试在device1的流s0上启动核函数，这会失败
MyKernel<<<100, 64, 0, s0>>>();

```

### 3.5.4 对等内存（Peer-to-Peer Memory ）

对等内存指一个device上的核函数可以直接读写另一个device的内存（需要硬件支持且运行在X64环境下）

检查方法：`cudaDeviceCanAccessPeer()`

```cpp
int canAccessPeer;
cudaDeviceCanAccessPeer(&canAccessPeer, device1, device2);
if (canAccessPeer) {
    // 设备间支持对等内存访问
}
```

启用方法：`cudaDeviceEnablePeerAccess()`

```cpp
cudaSetDevice(device1);
cudaDeviceEnablePeerAccess(device2, 0); // 启用device1对device2的对等访问
```

在非NVSwitch系统中，每个设备最多支持八个对等连接。

在开启对等内存访问的情况下，两个设备共享一个统一的地址空间（Unified Virtual Address Space）。这意味着一个指针可以同时用于访问两个设备的内存，无需进行额外的地址转换

```cpp
cudaSetDevice(0);  // 设置当前设备为设备0
float* p0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&p0, size);  // 在设备0上分配内存

MyKernel<<<1000, 128>>>(p0);  // 在设备0上启动核函数

cudaSetDevice(1);  // 设置当前设备为设备1
cudaDeviceEnablePeerAccess(0, 0);  // 启用设备1对设备0的对等访问

MyKernel<<<1000, 128>>>(p0);  // 在设备1上启动核函数，可以访问设备0上的内存地址p0

```

同样的，此时可以直接进行使用`cudaMemcpy`、`cudaMemcpyAsync`等完成内存复制

对于不适用统一地址空间的情况，需要使用专用的对等内存复制函数。包括 `cudaMemcpyPeer()`、`cudaMemcpyPeerAsync()`、`cudaMemcpy3DPeer()` 和 `cudaMemcpy3DPeerAsync()` 等

统一地址空间（Unified Virtual Address Space）的范围并不仅局限于device之间，其也包括host的内存空间。也就是说，在硬件支持的前提下，主机和设备将共享一个统一的虚拟地址空间。检查方法如下：`unifiedAddressing`=1。在UVAS环境下，将享受一下几点特性：

- **指针位置确定**：
    - 任何通过CUDA分配的主机内存或在使用统一地址空间的设备上分配的设备内存，其位置都可以通过 `cudaPointerGetAttributes()` 确定。这使得在调试和内存管理时可以方便地知道指针所指向的内存位置。
- **内存复制**：
    - 在进行内存复制操作时，如果使用统一地址空间，可以将 `cudaMemcpy*()` 的 `cudaMemcpyKind` 参数设置为 `cudaMemcpyDefault`。这样CUDA会根据指针自动确定内存位置并执行正确的复制操作。这对于非CUDA分配的主机内存也是适用的，只要当前设备使用统一地址空间。
- **可移植内存分配**：
    - 通过 `cudaHostAlloc()` 进行的主机内存分配在所有使用统一地址空间的设备之间是自动可移植的。通过 `cudaHostAlloc()` 返回的指针可以直接在这些设备上运行的内核中使用，不需要额外的设备指针转换。
- **查询统一地址空间支持**：
    - 可以通过检查设备属性 `unifiedAddressing` 是否等于1，来判断特定设备是否支持统一地址空间。这个属性可以在设备枚举过程中查询。

Tip：Linux环境下，当IOMMU启用时，无法直接使用Peer-to-Peer Memory，需要在虚拟机环境下

## 3.6 进程通信（Interprocess Communication）

进程间通信（Interprocess Communication，IPC）允许不同的进程共享设备内存和事件句柄，以实现跨进程的数据共享和同步。通过IPC，多个进程可以高效地共享GPU资源，而无需通过主机内存进行数据传递，从而提升性能。

- **同一进程内**：在同一进程中，任何线程都可以直接引用由其他线程创建的设备内存指针或事件句柄。
- **不同进程间**：不同进程中的线程无法直接引用彼此的设备内存指针或事件句柄，必须使用IPC API来共享这些资源。

Tip：不支持通过 `cudaMallocManaged` 分配的内存，且必须在X64的Linux环境下

- **cudaIpcGetMemHandle**：获取设备内存指针的IPC句柄。
- **cudaIpcOpenMemHandle**：打开IPC句柄，从中获取设备内存指针。
- **cudaIpcCloseMemHandle**：关闭通过IPC句柄打开的设备内存指针。
- **cudaIpcGetEventHandle**：获取事件的IPC句柄。
- **cudaIpcOpenEventHandle**：打开事件的IPC句柄，从中获取事件句柄。
1. **获取IPC句柄**：
    - 进程A在设备上分配内存，并通过 `cudaIpcGetMemHandle()` 获取该内存的IPC句柄。
    - 进程A通过某种方式（如文件或共享内存）将IPC句柄传递给进程B。
2. **打开IPC句柄**：
    - 进程B接收到IPC句柄后，通过 `cudaIpcOpenMemHandle()` 打开该句柄，从而获取设备内存指针。
3. **使用共享内存**：
    - 进程B可以直接使用这个设备内存指针进行读写操作，如同它是在本进程中分配的一样。
4. **关闭IPC句柄**：
    - 在不再需要使用共享内存时，进程B可以通过 `cudaIpcCloseMemHandle()` 关闭该内存指针。

## 3.7 错误检查

对所有的 CUDA Runtime 函数，其运行结束后都会返回一个错误代码，但对于异步函数，这个错误代码无法报告设备上发生的异步错误，因为函数在设备完成任务之前就返回了。

所有的 CUDA 运行时函数都会返回一个错误代码，但对于异步函数（如异步并发执行中的函数），这个错误代码无法报告设备上发生的异步错误，因为函数在设备完成任务之前就返回了（通常情况下，该错误代码返回的是在host上的因为参数验证错误引发的错误）。

检查方法：在异步函数调用之后立即进行同步操作，同步之后，可以检查 `cudaDeviceSynchronize()` 返回的错误代码来确定是否发生了异步错误。

对 Kernel 函数而言，其不会返回任何错误代码，因此必须在内核启动后调用 `cudaPeekAtLastError()` 或 `cudaGetLastError()` 来检索任何预启动错误。
为了确保 `cudaPeekAtLastError()` 或 `cudaGetLastError()` 返回的错误不来源于内核启动之前的调用，需要在内核启动前将运行时错误变量设置为 `cudaSuccess`，例如在内核启动前调用一次 `cudaGetLastError()`。由于内核启动是异步的，因此在调用 `cudaPeekAtLastError()` 或 `cudaGetLastError()` 之前，应用程序必须在内核启动和错误检查之间进行同步操作

下面给出一个简单的示例

```cpp
#include <iostream>
#include <cuda_runtime.h>

// 简单的内核函数
__global__ void simpleKernel(float* d_data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d_data[idx] = idx;
}

int main() {
    // 设置设备
    cudaSetDevice(0);

    // 分配设备内存
    float* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));

    // 在内核启动之前检查并重置错误变量
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error before kernel launch: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 启动内核
    simpleKernel<<<1, 1024>>>(d_data);

    // 检查内核启动后的错误
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error during kernel launch: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 同步设备以检查异步错误
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error after cudaDeviceSynchronize: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Kernel executed successfully." << std::endl;

    // 释放设备内存
    cudaFree(d_data);

    return 0;
}

```

## 3.8 调用栈 Call Stack

在CUDA中，调用栈是用于存储函数调用信息（如函数参数、本地变量和返回地址）的内存区域。对于递归调用或深度嵌套的函数调用，调用栈的大小可能需要调整。

- **查询调用栈大小**：
    - 使用 `cudaDeviceGetLimit()` 可以查询当前设备的调用栈大小。该函数接受两个参数：一个指向存储结果的变量的指针和一个限制类型（在这里为 `cudaLimitStackSize`）。
    
    ```cpp
    size_t stackSize;
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    std::cout << "Current call stack size: " << stackSize << " bytes" << std::endl;
    ```
    
- **设置调用栈大小**：
    - 使用 `cudaDeviceSetLimit()` 可以设置设备的调用栈大小。该函数接受两个参数：限制类型（`cudaLimitStackSize`）和新的大小值。
    
    ```cpp
    size_t newStackSize = 1024 * 1024; // 设置调用栈大小为1MB
    cudaDeviceSetLimit(cudaLimitStackSize, newStackSize);
    ```
    

## 3.9 CUDA 用户对象

CUDA 用户对象用于帮助管理 CUDA 中异步工作所使用的资源的生命周期，尤其体现在对 CUDA Graphs和流捕获（stream capture）这两个概念上。

CUDA 图不兼容与各种资源管理方案并不兼容，因为资源的指针或句柄是不固定的，需要间接引用或图更新，每次提交工作时都需要同步 CPU 代码。CUDA 用户对象提供了另一种方法。CUDA 用户对象将用户指定的析构函数回调与内部引用计数相关联，类似于 C++ 中的 `shared_ptr`。引用可以由 CPU 上的用户代码和 CUDA 图拥有。注意，对于用户拥有的引用，不像 C++ 智能指针，没有表示引用的对象；用户必须手动跟踪用户拥有的引用。一个典型的用例是在创建用户对象后立即将唯一的用户拥有引用移交给 CUDA 图。

当一个引用与 CUDA 图关联时，CUDA 将自动管理图操作。一个克隆的 `cudaGraph_t` 保留源 `cudaGraph_t` 所拥有的每个引用的副本，且具有相同的数量。一个实例化的 `cudaGraphExec_t` 保留源 `cudaGraph_t` 中每个引用的副本。当一个 `cudaGraphExec_t` 在没有同步的情况下被销毁时，引用会被保留直到执行完成。

```cpp
// 示例使用
cudaGraph_t graph; // 预先存在的图
Object *object = new Object; // C++ 对象，可能具有非平凡的析构函数
cudaUserObject_t cuObject;
cudaUserObjectCreate(
    &cuObject,
    object, // 这里我们使用一个 CUDA 提供的模板包装器来调用这个 API，
            // 它提供了删除 C++ 对象指针的回调
    1, // 初始引用计数
    cudaUserObjectNoDestructorSync // 确认回调不能通过 CUDA 等待
);
cudaGraphRetainUserObject(
    graph,
    cuObject,
    1, // 引用的数量
    cudaGraphUserObjectMove // 转移一个调用者拥有的引用（不修改总引用计数）
);
// 此线程不再拥有引用；不需要调用释放 API
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0); // 将保留一个新的引用
cudaGraphDestroy(graph); // graphExec 仍然拥有一个引用
cudaGraphLaunch(graphExec, 0); // 异步启动可以访问用户对象
cudaGraphExecDestroy(graphExec); // 启动未同步；如果需要，释放将被延迟
cudaStreamSynchronize(0); // 在启动同步后，剩余的引用被释放并执行析构函数
                          // 注意这是异步发生的。
                          // 如果析构回调信号了一个同步对象，现在等待它是安全的。

```

图中子图节点所拥有的引用与子图关联，而不是与父图关联。如果一个子图被更新或删除，引用会相应更改。如果使用 `cudaGraphExecUpdate` 或 `cudaGraphExecChildGraphNodeSetParams` 更新一个可执行图或子图，新源图中的引用会被克隆并替换目标图中的引用。在任何情况下，如果以前的启动未同步，任何将被释放的引用将被保留直到启动完成执行。

# 4. 资产管理

## 4.1 图形资产 互操作性

某些来自 OpenGL 和 Direct3D 的资源可以映射到 CUDA 的地址空间，从而允许 CUDA 读取 OpenGL 或 Direct3D 写入的数据，或让 CUDA 写入的数据供 OpenGL 或 Direct3D 使用。

在资源映射之前，必须先使用 OpenGL 互操作性和 Direct3D 互操作性中提到的函数将资源注册到 CUDA。这些函数会返回一个指向 `struct cudaGraphicsResource` 类型的 CUDA 图形资源的指针。由于注册资源可能会有较高的开销，因此通常每个资源只注册一次。可以使用 `cudaGraphicsUnregisterResource()` 取消注册 CUDA 图形资源。每个想要使用该资源的 CUDA 上下文都需要单独注册该资源。

一旦资源注册到 CUDA 后，可以多次使用 `cudaGraphicsMapResources()` 和 `cudaGraphicsUnmapResources()` 进行映射和取消映射。可以调用 `cudaGraphicsResourceSetMapFlags()` 来指定使用提示（如只写或只读），以优化 CUDA 驱动程序的资源管理。

内核可以通过 `cudaGraphicsResourceGetMappedPointer()` 返回的设备内存地址读取或写入已映射的资源，而对于 CUDA 数组，则使用 `cudaGraphicsSubResourceGetMappedArray()`。在资源映射期间，通过 OpenGL、Direct3D 或其他 CUDA 上下文访问该资源会产生未定义的结果。OpenGL 互操作性和 Direct3D 互操作性提供了每个图形 API 的具体信息和一些代码示例。SLI 互操作性则提供了系统处于 SLI 模式时的具体信息。

### 4.1.1 OpenGL Interoperability

来自 OpenGL 的缓冲区、纹理和渲染缓冲区对象可以映射到 CUDA 的地址空间。这些资源可以通过以下方式注册并在 CUDA 中使用：

1. **缓冲区对象（Buffer Object）**：
    - 使用 `cudaGraphicsGLRegisterBuffer()` 注册。
    - 在 CUDA 中，它表现为一个设备指针，可以被内核读写，也可以通过 `cudaMemcpy()` 进行操作。
2. **纹理或渲染缓冲区对象（Texture or Renderbuffer Object）**：
    - 使用 `cudaGraphicsGLRegisterImage()` 注册。
    - 在 CUDA 中，它表现为一个 CUDA 数组。
    - 内核可以通过将其绑定到纹理或表面引用来读取该数组。
    - 如果资源注册时使用了 `cudaGraphicsRegisterFlagsSurfaceLoadStore` 标志，内核也可以通过表面写函数写入该数组。
    - 该数组也可以通过 `cudaMemcpy2D()` 调用进行读写。
    - `cudaGraphicsGLRegisterImage()` 支持具有 1、2 或 4 个组件且内部类型为浮点（如 GL_RGBA_FLOAT32）、归一化整数（如 GL_RGBA8、GL_INTENSITY16）和非归一化整数（如 GL_RGBA8UI）的所有纹理格式（请注意，由于非归一化整数格式需要 OpenGL 3.0，因此只能由着色器写入，不能由固定功能管线写入）。
3. **OpenGL 上下文**：
    - 要共享资源的 OpenGL 上下文必须是当前主机线程的上下文，以进行任何 OpenGL 互操作性 API 调用。
4. **Bindless 纹理**：
    - 当 OpenGL 纹理变为无绑定状态（例如通过使用 `glGetTextureHandle*` 或 `glGetImageHandle*` API 请求图像或纹理句柄）时，不能将其注册到 CUDA。应用程序需要在请求图像或纹理句柄之前注册纹理以进行互操作。

```cpp
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>

GLuint positionsVBO;
struct cudaGraphicsResource* positionsVBO_CUDA;
int width = 1024;
int height = 1024;
float time = 0.0f;

// CUDA核函数，创建顶点数据
__global__ void createVertices(float4* positions, float time, unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 计算uv坐标
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;
    
    // 计算简单的正弦波模式
    float freq = 4.0f;
    float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;
    
    // 写入顶点位置
    positions[y * width + x] = make_float4(u, w, v, 1.0f);
}

// 显示函数
void display()
{
    // 映射缓冲区对象以便从CUDA写入
    float4* positions;
    cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, positionsVBO_CUDA);
    
    // 执行CUDA核函数
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
    createVertices<<<dimGrid, dimBlock>>>(positions, time, width, height);
    
    // 取消映射缓冲区对象
    cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);
    
    // 从缓冲区对象渲染
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, width * height);
    glDisableClientState(GL_VERTEX_ARRAY);
    
    // 交换缓冲区
    glutSwapBuffers();
    glutPostRedisplay();
}

// 删除VBO
void deleteVBO()
{
    cudaGraphicsUnregisterResource(positionsVBO_CUDA);
    glDeleteBuffers(1, &positionsVBO);
}

int main(int argc, char** argv)
{
    // 初始化OpenGL和GLUT以使用设备0，并使OpenGL上下文成为当前上下文
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA-OpenGL Interop");
    glewInit();
    
    // 设置显示回调函数
    glutDisplayFunc(display);
    
    // 显式设置设备0
    cudaSetDevice(0);
    
    // 创建缓冲区对象并注册到CUDA
    glGenBuffers(1, &positionsVBO);
    glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
    unsigned int size = width * height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, positionsVBO, cudaGraphicsMapFlagsWriteDiscard);
    
    // 启动渲染循环
    glutMainLoop();
    
    // 清理资源
    deleteVBO();
    
    return 0;
}

```

在 Windows 系统上，对于 Quadro GPU，可以使用 `cudaWGLGetDevice()` 来获取与 `wglEnumGpusNV()` 返回的句柄相关联的 CUDA 设备。在多 GPU 配置中，Quadro GPU 提供比 GeForce 和 Tesla GPU 更高性能的 OpenGL 互操作性，其中 OpenGL 渲染在 Quadro GPU 上执行，而 CUDA 计算在系统中的其他 GPU 上执行。

### 4.1.2 Direct3D Interoperability

Direct3D 互操作性支持 Direct3D 9Ex、Direct3D 10 和 Direct3D 11。一个 CUDA 上下文只能与满足以下条件的 Direct3D 设备进行互操作：

- **Direct3D 9Ex 设备**：必须使用 `DeviceType` 设置为 `D3DDEVTYPE_HAL` 并且 `BehaviorFlags` 设置为 `D3DCREATE_HARDWARE_VERTEXPROCESSING` 标志创建。
- **Direct3D 10 和 Direct3D 11 设备**：必须使用 `DriverType` 设置为 `D3D_DRIVER_TYPE_HARDWARE` 创建。

可以映射到 CUDA 地址空间的 Direct3D 资源包括 Direct3D 缓冲区、纹理和表面。这些资源通过以下函数注册：

- `cudaGraphicsD3D9RegisterResource()`
- `cudaGraphicsD3D10RegisterResource()`
- `cudaGraphicsD3D11RegisterResource()`

以下代码示例使用一个内核动态修改存储在顶点缓冲区对象中的宽度 x 高度的二维网格顶点。（仅给出Direct 3D 11版本的示例）

```cpp
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <d3d11.h>
#include <dxgi.h>
#include <iostream>

ID3D11Device* device;
struct CUSTOMVERTEX {
    FLOAT x, y, z;
    DWORD color;
};
ID3D11Buffer* positionsVB;
struct cudaGraphicsResource* positionsVB_CUDA;
int width = 1024;
int height = 1024;
float time = 0.0f;

int main()
{
    int dev;
    // 获取支持CUDA的适配器
    IDXGIFactory* factory;
    CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory);
    IDXGIAdapter* adapter = 0;
    for (unsigned int i = 0; !adapter; ++i) {
        if (FAILED(factory->EnumAdapters(i, &adapter)))
            break;
        if (cudaD3D11GetDevice(&dev, adapter) == cudaSuccess)
            break;
        adapter->Release();
    }
    factory->Release();
    
    // 创建交换链和设备
    // 省略具体创建代码
    
    // 使用相同的设备
    cudaSetDevice(dev);
    
    // 创建顶点缓冲区并注册到CUDA
    unsigned int size = width * height * sizeof(CUSTOMVERTEX);
    D3D11_BUFFER_DESC bufferDesc;
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth = size;
    bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bufferDesc.CPUAccessFlags = 0;
    bufferDesc.MiscFlags = 0;
    device->CreateBuffer(&bufferDesc, 0, &positionsVB);
    cudaGraphicsD3D11RegisterResource(&positionsVB_CUDA, positionsVB, cudaGraphicsRegisterFlagsNone);
    cudaGraphicsResourceSetMapFlags(positionsVB_CUDA, cudaGraphicsMapFlagsWriteDiscard);
    
    // 启动渲染循环
    while (...) {
        // 省略具体渲染代码
        Render();
    }
    
    // 释放资源
    releaseVB();
    
    return 0;
}

void Render()
{
    // 映射顶点缓冲区以便从CUDA写入
    float4* positions;
    cudaGraphicsMapResources(1, &positionsVB_CUDA, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, positionsVB_CUDA);
    
    // 执行CUDA核函数
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
    createVertices<<<dimGrid, dimBlock>>>(positions, time, width, height);
    
    // 取消映射顶点缓冲区
    cudaGraphicsUnmapResources(1, &positionsVB_CUDA, 0);
    
    // 绘制并呈现
    // 省略具体绘制代码
}

void releaseVB()
{
    cudaGraphicsUnregisterResource(positionsVB_CUDA);
    positionsVB->Release();
}

__global__ void createVertices(float4* positions, float time, unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 计算uv坐标
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;
    
    // 计算简单的正弦波模式
    float freq = 4.0f;
    float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;
    
    // 写入顶点位置
    positions[y * width + x] = make_float4(u, w, v, __int_as_float(0xff00ff00));
}

```

### 4.1.3 SLI Interoperability

在多 GPU 系统中，所有支持 CUDA 的 GPU 都可以通过 CUDA 驱动程序和运行时作为独立设备访问。然而，当系统处于 SLI 模式时，需要注意以下特殊考虑。

首先，在一个 GPU 上的一个 CUDA 设备上的内存分配将会消耗属于 Direct3D 或 OpenGL 设备的 SLI 配置中的其他 GPU 上的内存。因此，分配可能会比预期更早失败。

其次，应用程序应为 SLI 配置中的每个 GPU 创建多个 CUDA 上下文。虽然这不是一个严格要求，但它可以避免设备之间不必要的数据传输。应用程序可以使用 `cudaD3D[9|10|11]GetDevices()`（针对 Direct3D）和 `cudaGLGetDevices()`（针对 OpenGL）调用来识别当前和下一个帧中执行渲染的设备的 CUDA 设备句柄。根据这些信息，应用程序通常会选择合适的设备，并将 Direct3D 或 OpenGL 资源映射到由 `cudaD3D[9|10|11]GetDevices()` 或 `cudaGLGetDevices()` 返回的 CUDA 设备上，当 `deviceList` 参数设置为 `cudaD3D[9|10|11]DeviceListCurrentFrame` 或 `cudaGLDeviceListCurrentFrame` 时。

请注意，从 `cudaGraphicsD9D[9|10|11]RegisterResource` 和 `cudaGraphicsGLRegister[Buffer|Image]` 返回的资源必须仅在注册发生的设备上使用。因此，在 SLI 配置中，当不同帧的数据在不同的 CUDA 设备上计算时，需要分别为每个资源注册。有关 CUDA 运行时如何与 Direct3D 和 OpenGL 互操作的详细信息，请参见 Direct3D 互操作性和 OpenGL 互操作性部分。

## 4.2 外部资源 互操作性

外部资源互操作性允许 CUDA 导入由其他 API 明确导出的某些资源。这些对象通常由其他 API 使用操作系统原生的句柄导出，例如在 Linux 上的文件描述符或在 Windows 上的 NT 句柄。它们也可以通过其他统一接口（如 NVIDIA 软件通信接口）导出。可以导入的资源有两种类型：内存对象和同步对象。

**内存对象**：

- 可以使用 `cudaImportExternalMemory()` 导入内存对象。
- 导入的内存对象可以通过内核中的设备指针访问，这些指针通过 `cudaExternalMemoryGetMappedBuffer()` 映射到内存对象，或者通过 `cudaExternalMemoryGetMappedMipmappedArray()` 映射到 CUDA mipmapped 数组。
- 根据内存对象的类型，可能在一个内存对象上设置多个映射。映射必须与导出 API 中设置的映射匹配，否则会导致未定义的行为。
- 导入的内存对象必须使用 `cudaDestroyExternalMemory()` 释放。释放内存对象不会释放与该对象的任何映射。因此，任何映射到该对象的设备指针必须使用 `cudaFree()` 显式释放，任何映射到该对象的 CUDA mipmapped 数组必须使用 `cudaFreeMipmappedArray()` 显式释放。销毁对象后访问其映射是非法的。

**同步对象**：

- 可以使用 `cudaImportExternalSemaphore()` 导入同步对象。
- 导入的同步对象可以使用 `cudaSignalExternalSemaphoresAsync()` 发出信号，并使用 `cudaWaitExternalSemaphoresAsync()` 等待信号。发出等待信号之前，必须先发出相应的信号。
- 根据导入的同步对象的类型，可能对如何发出信号和等待信号有额外的限制，这些限制将在后续部分中描述。
- 导入的信号量对象必须使用 `cudaDestroyExternalSemaphore()` 释放。在销毁信号量对象之前，所有未完成的信号和等待必须已经完成。

### 4.2.1 Vulkan

### 4.2.2 OpenGL

### 4.2.3 Direct3D

### 4.2.4 NVIDIA Software Communication Interface Interoperability (NVSCI)

# 5. 并行的硬件实现

NVIDIA GPU 基于多线程流式多处理器 (SM: Streaming Multiprocessors)构建。当主机 CPU 上的 CUDA 程序调用内核网格时，Grid 的 Block 会被枚举并分配给具有可用执行能力的多处理器。一个Block的Threads在一个多处理器上并发执行，并且多个线程块可以在一个多处理器上并发执行。当线程块终止时，新块会在腾出的多处理器上启动。

## 5.1 SIMT Architecture

SM 以 32 个并行线程组（称为 Warp）的形式来创建、管理、调度和执行线程。在CUDA的SIMT架构中，warp中的所有线程从同一个程序地址（即同一个程序计数器值）开始执行同一条指令。然而，每个线程都有自己独立的程序计数器和寄存器状态，这意味着尽管所有线程最初从同一个地址开始，它们可以根据条件分支等因素独立地分支和执行不同的指令路径。

当SM被分配一个或多个线程块进行执行时，它会将它们分成多个warp，每个warp由一个warp调度器进行调度执行。对于单个warp，其内部的线程ID（thread ID）是连续增加的，即第一个warp包含线程ID为0到31的线程，第二个warp包含线程ID为32到63的线程，以此类推。

一个Warp一次执行一条公共指令，易知，当warp中的所有32个线程执行完全相同的指令路径时，效率最高。如果warp中的线程遇到条件分支（如 `if-else` 语句），并且不同线程的条件判断结果不同，线程就会分歧。当发生分歧时，warp会分别执行每个分支路径。对于当前不在执行路径上的线程，会被暂时禁用。举例来说，如果有一半的线程满足条件进入 `if` 分支，另一半进入 `else` 分支，warp会先执行 `if` 分支，然后执行 `else` 分支。在执行 `if` 分支时，属于 `else` 分支的线程会被禁用，反之亦然。分歧（Divergence）只在warp内发生。不同的warp可以同时执行不同的指令，无论它们是否在执行相同的代码路径

SIMT 体系结构类似于 SIMD（单指令多数据）向量组成 (Organization)，其中由单指令控制多个处理元素。SIMT与SIMD的主要区别在于，在SIMD架构中，一条指令同时操作多个数据元素。SIMD处理器使用宽向量寄存器来存储多个数据元素，这些寄存器的宽度（也就是能存储的数据元素个数）是固定的并且对软件是可见的。而SIMT架构中每个线程都有自己的指令地址计数器和寄存器状态。这意味着每个线程可以独立地执行和分支，而不需要程序员手动管理“宽度”。

与SIMD相比，SIMT允许程序员为独立的标量线程编写线程级并行代码，也为协调线程编写数据并行代码。为确保正确性，程序员基本上可以忽略SIMT行为；然而，通过确保代码很少需要warp中的线程发生分歧，可以实现显著的性能提升。在实践中，这类似于传统代码中的缓存行：设计正确性时可以安全地忽略缓存行大小，但设计峰值性能时必须考虑代码结构。另一方面，矢量架构要求软件将加载合并成矢量并手动管理分歧

在NVIDIA Volta之前，warp使用单个程序计数器，由warp的所有32个线程共享，并有一个活动掩码指定warp的活动线程。因此，来自同一warp的线程在分歧区域或不同执行状态下无法互相信号或交换数据，需要细粒度数据共享并由锁或互斥量保护的算法可能会导致死锁，具体取决于争用线程来自哪个warp。

从NVIDIA Volta架构开始，独立线程调度允许线程之间完全并发，不受warp限制。通过独立线程调度，GPU为每个线程维护执行状态，包括程序计数器和调用栈，并可以在每个线程的粒度上让出执行，既可以更好地利用执行资源，也可以让一个线程等待另一个线程产生数据。调度优化器决定如何将同一warp中的活动线程分组到SIMT单元中。这样既保留了之前NVIDIA GPU的高吞吐量SIMT执行，同时具有更大的灵活性：线程现在可以在亚warp粒度上分歧和重新汇合。

独立线程调度可能导致参与执行代码的线程与开发者假定的warp同步性不同。特别是，任何warp同步代码（如无同步的warp内归约）都应重新审视以确保与NVIDIA Volta及更高版本的兼容性。有关更多详细信息，请参见计算能力7.x。

**tip：**

执行当前指令的warp线程称为活动线程，而不在当前指令上的线程称为非活动（禁用）线程。线程可能由于各种原因处于非活动状态，包括比warp中的其他线程更早退出，采取与warp当前执行路径不同的分支路径，或块的最后几个线程的数量不是warp大小的倍数。

如果warp执行的非原子指令写入全局或共享内存中的同一位置，warp中的多个线程对该位置进行写入，发生的序列化写入次数取决于设备的计算能力（参见计算能力3.x、计算能力5.x、计算能力6.x和计算能力7.x），执行最终写入的线程未定义。

如果warp执行的原子指令读取、修改并写入全局内存中的同一位置，warp中的多个线程对该位置进行读/修改/写，每次操作都会发生且序列化，但发生的顺序未定义。

## 5.2 硬件多线程 Hardware Multithreading

在SM上处理的每个warp的执行上下文（包括程序计数器、寄存器等）在其整个生命周期内都保存在片上。因此，切换执行上下文并不会带来性能开销。在每个指令发出时，warp调度器会选择一个准备好执行下一条指令的warp（即该warp的活动线程）并向这些线程发出指令。

具体来说，每个SM有一组32位寄存器，这些寄存器在warp之间分配，同时并行数据缓存 (parallel data cache) 和共享内存 (shared memory) 被划分到不同的线程块之间。

一个给定的内核在SM上可以驻留和处理的块和warp的数量，取决于内核使用的寄存器和共享内存的数量以及多处理器上可用的寄存器和共享内存的数量。每个多处理器还有一个最大驻留块数和最大驻留warp数。这些限制以及多处理器上可用的寄存器和共享内存的数量是设备计算能力的函数，并在计算能力中给出。如果每个多处理器上没有足够的寄存器或共享内存来处理至少一个块，则内核将无法启动。

一个块中的warp总数如下：

$$
 ceil(\frac{T}{W_{size}},1) 
$$

- T 是每个块中的线程数
- 𝑊𝑠𝑖𝑧𝑒 是 Warp 的大小，默认为32
- 𝑐𝑒𝑖𝑙(𝑥,𝑦) 是 x 四舍五入到 y 的整数倍。

为每个块分配的寄存器总数以及共享内存总量记录在 CUDA 工具包提供的 CUDA Occupancy Calculator中

# 6. 性能优化

## 6.1 整体性能优化策略

性能优化围绕四个基本策略展开：

1. **最大化并行执行以实现最大利用率**；
2. **优化内存使用以实现最大内存吞吐量**；
3. **优化指令使用以实现最大指令吞吐量**；
4. **最小化内存抖动**。

策略的选择具体取决于应用程序在具体情况下的性能短板，例如，对于主要受内存访问限制的内核，优化指令使用不会带来显著的性能提升。因此，优化工作应该始终通过测量和监控性能限制因素来指导，例如使用 CUDA 分析器。此外，将特定内核的浮点运算吞吐量或内存吞吐量（取决于哪一个更有意义）与设备的对应理论峰值吞吐量进行比较，可以指示该内核还有多少改进空间。

## 6.2 并行优化

为了最大化利用率，应用程序应该结构化，以便暴露尽可能多的并行性，并有效地将这种并行性映射到系统的各个组件上，使它们大部分时间都处于忙碌状态。

### 6.2.1 应用层面

在宏观层次上，应用程序应该使用异步函数调用和流（如在异步并发执行中描述的）来最大化主机、设备、以及将主机与设备相连的总线之间的并行执行。同时应用程序根据工作类型将任务进行分配：：串行任务发送给主机；将并行任务发送给设备。

对于并行工作负载，在算法中某些线程需要同步以便共享数据的点，有两种情况：

- 如果这些线程属于同一个Block，它们应该使用 `__syncthreads()` 并通过共享内存在同一核函数调用中共享数据。
- 如果这些线程属于不同的Block，它们必须通过全局内存在两个单独的核函数调用中共享数据，一个用于写入全局内存，一个用于从全局内存读取数据。
    - 因为此方式使用了额外的核函数调用和全局内存通信，其效率较低

因此，应该通过将算法映射到CUDA编程模型，使需要线程间通信的计算尽可能在单个线程块内执行，来最小化这种情况的发生。

### 6.2.2 设备层面

在次低的层面上，应用程序应该最大化设备中 SM 之间的并行执行。

多个核函数可以在一个设备上同时执行，因此也可以通过使用 Stream 来启用足够多的内核来实现设备的利用率最大化，如[异步并发执行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)中所述。

### 6.2.3 SM层面

在最低层次上，应用程序应该在多处理器内部的各个功能单元之间最大化并行执行。

**利用率与warp数量**

- 线程级并行性**：**GPU多处理器主要依靠线程级并行性来最大化其功能单元的利用率。因此，利用率直接与驻留warp的数量相关。
- 指令发射与延迟隐藏：
    - 每当指令发射时，warp调度器会选择一条准备好执行的指令。该指令可以是相同warp中的另一条独立指令（指令级并行性），也可以是另一warp的指令（线程级并行性）。
    - 一旦指令被选择好，就会被发射到warp的活动线程中。warp准备好执行下一条指令所需的时钟周期数称为延迟（Latency）。
    - 如果所有warp调度器在延迟期间的每个时钟周期上都有一些可以发射的指令，那么GPU就可以实现完全利用，即延迟被完全隐藏。
    - 隐藏延迟所需的指令数量取决于这些指令的吞吐量（各种算术指令的吞吐量见算术指令）。假设指令具有最大吞吐量，它等于：
        - 4L：计算能力为5.x、6.1、6.2、7.x和8.x的设备：多处理器在一个时钟周期内为四个warp中的每个warp发出一条指令
        - 2L：计算能力为6.0的设备：每个时钟周期发出的两条指令分别来自两个不同的warp。
        - 8L：对算能力为3.x的设备：每个时钟周期发出的八条指令是四对指令，分别来自四个不同的warp，每对指令来自同一个warp。
- 操作数存储位置对Warp的影响
    - **寄存器依赖性**：
        - 当所有输入操作数都在寄存器中时，延迟是由寄存器依赖性引起的，即一些输入操作数是由之前的指令写入的，而这些指令尚未执行完毕。
        - 在这种情况下，延迟等于前一指令的执行时间，warp调度器必须在这段时间内调度其他warp的指令。
        - 例如，在计算能力为7.x的设备上，大多数算术指令的执行时间通常是4个时钟周期。为了隐藏算术指令的延迟，每个多处理器需要16个活动warp（因为4个时钟周期和4个warp调度器）。
    - 片外内存依赖性
        - 当某些输入操作数位于片外内存（即主存）中时，延迟会大大增加，通常达到数百个时钟周期。
        - 为了在这种高延迟期间保持warp调度器的忙碌，需要更多的warp来填补这个时间间隙。需求的warp数量取决于内核代码和其指令级并行性的程度。
- 内存栅栏和同步点对Warp数量
    - **同步点等待**：在同步点处等待会导致多处理器闲置，因为同一块中的其他warp还没有完成同步点之前的指令。
    - **减少闲置**：通过在每个多处理器上驻留多个块，可以减少这种闲置，因为不同块中的warp不需要在同步点相互等待。
- 驻留块和warp数量的影响因素
    - **执行配置**：在调用内核时指定的执行配置（例如每个块的线程数和总块数）影响多处理器上可以驻留的块和warp的数量。
    - **多处理器的内存资源**：每个多处理器可用的寄存器和共享内存数量限制了它能同时容纳多少块和warp。
    - **内核的资源需求**：内核使用的寄存器和共享内存数量直接影响多处理器上能容纳的块和warp数量。
- 寄存器使用对驻留warp数量
    - 寄存器数量
        - 对于计算能力为6.x的设备，如果一个内核使用64个寄存器，并且每个块有512个线程，并且需要非常少的共享内存，那么两个块（即32个warp）可以驻留在多处理器上，因为它们需要2x512x64个寄存器，这正好匹配多处理器上可用的寄存器数量。
        - 但是，如果内核使用的寄存器数量增加到65个，则只能驻留一个块（即16个warp），因为两个块需要2x512x65个寄存器，这超过了多处理器上可用的寄存器数量。
        - 寄存器文件是以32位寄存器组织的，因此每个存储在寄存器中的变量至少需要一个32位寄存器。例如，一个双精度变量需要两个32位寄存器。
    - 编译器优化
        - 编译器会尝试在最小化寄存器溢出（即当寄存器不足时使用慢速的全局内存）和指令数量的同时，尽量减少寄存器的使用。可以使用 `maxrregcount` 编译器选项或启动边界（Launch Bounds）来控制寄存器的使用。
        - 使用 `--ptxas-options=-v` 选项编译时，编译器会报告内核使用的寄存器和共享内存数量。
    
    块中的线程数量应选择为warp大小的倍数，以尽量避免因warp中线程不足而浪费计算资源。有几个 API 函数来帮助程序员根据寄存器和共享内存的大小要求选择适合的线程块大小。
    
    - 占用率计算
        1. **占用率计算器API（cudaOccupancyMaxActiveBlocksPerMultiprocessor）**：
            - 该API函数可以根据内核的块大小和共享内存使用情况提供占用率预测。它报告的是每个多处理器的并发线程块数量。
        2. **占用率转换**：
            - 报告的并发线程块数量可以转换为其他度量。将这个值乘以每块的warp数，得到每个多处理器的并发warp数；再将并发warp数除以每个多处理器的最大warp数，可以得到占用率的百分比。
        3. **基于占用率的启动配置API（cudaOccupancyMaxPotentialBlockSize和cudaOccupancyMaxPotentialBlockSizeVariableSMem）**：
            - 这些API函数通过启发式计算执行配置，以实现最大的多处理器级别的占用率。
        4. **占用率计算器API（cudaOccupancyMaxActiveClusters）**：
            - 该API函数可以根据集群大小、块大小和共享内存使用情况提供占用率预测。它报告的是系统中GPU上给定大小的最大活动集群数量。
        
        下面给出一个示例
        
        ```cpp
        // 计算 MyKernel 的占用率
        __global__ void MyKernel(int *d, int *a, int *b) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            d[idx] = a[idx] * b[idx];
        }
        int main() {
            int numBlocks;  // 活动块数
            int blockSize = 32;
            // 这些变量用于将占用率转换为warps
            int device;
            cudaDeviceProp prop;
            int activeWarps;
            int maxWarps;
        
            // 获取设备属性
            cudaGetDevice(&device);
            cudaGetDeviceProperties(&prop, device);
        
            // 计算每个多处理器的最大活动块数
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &numBlocks,
                MyKernel,
                blockSize,
                0
            );
        
            // 计算活动warps和最大warps数量
            activeWarps = numBlocks * blockSize / prop.warpSize;
            maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
        
            // 输出占用率
            std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
        
            return 0;
        }
        //根据用户输入配置基于占用率的内核启动
        __global__ void MyKernel(int *array, int arrayCount) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < arrayCount) {
                array[idx] *= array[idx];
            }
        }
        int launchMyKernel(int *array, int arrayCount) {
            int blockSize;    // 启动配置返回的块大小
            int minGridSize;  // 实现全设备最大占用率所需的最小网格大小
            int gridSize;     // 根据输入大小计算的实际网格大小
        
            // 计算实现最大占用率所需的块大小和最小网格大小
            cudaOccupancyMaxPotentialBlockSize(
                &minGridSize,
                &blockSize,
                (void*)MyKernel,
                0,
                arrayCount
            );
        
            // 根据数组大小向上取整
            gridSize = (arrayCount + blockSize - 1) / blockSize;
        
            // 启动内核
            MyKernel<<<gridSize, blockSize>>>(array, arrayCount);
            cudaDeviceSynchronize();
        
            return 0;
        }
        //使用集群占用率API查找最大活动集群数量
        int main() {
            cudaLaunchConfig_t config = {0};
            config.gridDim = number_of_blocks;
            config.blockDim = 128;  // 每块128个线程
            config.dynamicSmemBytes = dynamic_shared_memory_size;
        
            cudaLaunchAttribute attribute[1];
            attribute[0].id = cudaLaunchAttributeClusterDimension;
            attribute[0].val.clusterDim.x = 2;  // 集群大小为2
            attribute[0].val.clusterDim.y = 1;
            attribute[0].val.clusterDim.z = 1;
            config.attrs = attribute;
            config.numAttrs = 1;
        
            int max_cluster_size = 0;
            // 查询最大集群大小
            cudaOccupancyMaxPotentialClusterSize(&max_cluster_size, (void *)kernel, &config);
            int max_active_clusters = 0;
            // 查询最大活动集群数量
            cudaOccupancyMaxActiveClusters(&max_active_clusters, (void *)kernel, &config);
        
            std::cout << "Max Active Clusters of size 2: " << max_active_clusters << std::endl;
        
            return 0;
        }
        
        ```
        

## 6.3 内存优化

最大化应用程序的整体内存吞吐量的第一步是最小化低带宽的数据传输（通常来说，主机和设备之间的内存数据传输的带宽较小，容易出现性能瓶颈），也要尽量减少全局内存与设备之间的数据传输，最大化片上内存（共享内存和缓存）的使用，包括L1 缓存和 L2 缓存（在计算能力2.x及更高版本的设备上可用）、纹理缓存和常量缓存（在所有设备上可用）。

内核的内存访问吞吐量可能因不同类型内存的访问模式而变化一个数量级。因此，最大化内存吞吐量的下一步是根据设备内存访问中描述的最佳内存访问模式尽可能优化内存访问。这种优化对于全局内存访问尤其重要，因为与片上带宽和算术指令吞吐量相比，全局内存带宽较低，因此非最佳的全局内存访问通常会对性能产生较大影响。

### 6.3.1 主机和设备之间的内存数据传输

为了减少host与device之间的数据传输，可以采用以下几种策列：

1. 将代码迁移到device上
    1. 即使运行内核时没有充分的并行性，也可以将更多代码从主机移到设备上，以减少数据传输。
    2. 中间数据结构可以在设备内存中创建、操作和销毁，而无需映射到主机或复制到主机内存。
2. 批量传输数据
    1. 由于每次传输都有开销，将许多小传输合并为一个大传输总是比分别进行每次传输性能更好。
3. 使用页锁定主机内存
    1. 在具有前端总线的系统上，通过使用页锁定主机内存，可以实现主机和设备之间的数据传输的更高性能。
4. 使用映射的页锁定内存
    1. 使用映射的页锁定内存时，无需分配任何设备内存，也无需显式地在设备和主机内存之间复制数据。
    2. 每次内核访问映射内存时，数据传输会隐式进行。
    3. 为了获得最佳性能，这些内存访问必须像访问全局内存一样进行合并（参见设备内存访问）。
    4. 假设内存访问是合并的，并且映射内存仅被读取或写入一次，那么使用映射的页锁定内存代替在设备和主机内存之间的显式复制可以提高性能。
5. 统一内存系统中的传输
    1. 在设备内存和主机内存物理相同的集成系统中，任何在主机和设备内存之间的复制都是多余的，应该使用映射的页锁定内存。（参考apple）
    2. 应用程序可以通过检查设备的集成属性（参见设备枚举）是否等于1来查询设备是否为集成设备。

### 6.3.2 设备内存访问

CUDA环境中，主要包含以下几种类型的存储

1. 全局内存 Global Memory
    - **位置**：设备内存。
    - **特性**：大容量、高延迟、低带宽。
    - **用途**：所有线程可以访问，适合存储大量数据。
    - **访问方式**：需要合并访问来提高效率。
2. 常量内存 Constant Memory
    - **位置**：设备内存中的一部分，具有常量缓存。
    - **特性**：小容量、低延迟（缓存命中时）、高带宽（缓存命中时）。
    - **用途**：存储不经常改变的常量数据，所有线程都可以读取。
    - **访问方式**：通过常量缓存来加速访问，适用于只读数据。
3. 本地内存 Local Memory
    - **位置**：设备内存。
    - **特性**：高延迟、低带宽。
    - **用途**：用于存储超出寄存器容量的临时变量（寄存器溢出时）。
    - **访问方式**：每个线程私有，通常访问速度较慢。
4. 纹理与表面内存 Texture and Surface Memory
    - **位置**：设备内存中的一部分，具有缓存。
    - **特性**：低延迟（缓存命中时）、高带宽（缓存命中时）。
    - **用途**：优化2D和3D空间访问模式的数据，具有特殊的插值功能。
    - **访问方式**：通过缓存来加速访问，适用于复杂的数据访问模式。
5. 共享内存 Share Memory
    - **位置**：位于多处理器芯片上。
    - **特性**：低延迟、高带宽。
    - **用途**：同一线程块内的线程共享，可以实现线程间的快速数据交换。
    - **访问方式**：通过内存银行实现高效访问，避免银行冲突。
6. 寄存器 Register
    - **位置**：位于多处理器芯片上。
    - **特性**：最低延迟、最高带宽。
    - **用途**：每个线程私有，存储临时变量和局部变量。
    - **访问方式**：直接访问，速度最快。

一条访问可寻址内存（如全局、局部、共享、常量或纹理内存）的指令可能需要多次重定向，这取决于warp中各线程内存地址的分布。不同类型的内存将会对质量吞吐量有着不一样的影响

**内存事务**：在硬件层面上进行的数据传输，大小为32字节、64字节或128字节。内存事务用于将数据从全局内存加载到缓存或者从缓存写回到全局内存。这种事务必须自然对齐，即内存事务的起始地址必须是事务大小的倍数。

**内存指令**：这是程序级别的内存操作指令，支持的数据类型大小为1字节、2字节、4字节、8字节或16字节。这些指令在汇编层面上会被编译成访问全局内存的指令。

- 全局内存访问
    
    全局内存位于设备的内存中（VRAM），而设备内存通过内存事务进行访问。
    
    当一个warp执行访问全局内存的指令时，根据每个线程访问的字的大小和线程之间的内存地址分布，它将warp内线程的内存访问合并为一个或多个这样的内存事务。通常，所需的事务越多，除了线程访问的字之外传输的未使用字也越多，从而相应地减少指令的吞吐量。例如，如果每个线程的4字节访问生成一个32字节的内存事务，那么吞吐量将减少到原来的八分之一。
    
    所需的事务数量以及最终受到的吞吐量影响因设备的计算能力而异。计算能力3.x、计算能力5.x、计算能力6.x、计算能力7.x、计算能力8.x和计算能力9.0提供了有关各种计算能力下如何处理全局内存访问的更多详细信息。
    
    为了最大化全局内存吞吐量，重要的是要通过以下方式最大化合并：
    
    - 遵循基于计算能力3.x、计算能力5.x、计算能力6.x、计算能力7.x、计算能力8.x和计算能力9.0的最优访问模式，
    - 使用符合大小和对齐要求的数据类型（详见下文的大小和对齐要求部分），
    - 在某些情况下对数据进行填充，例如，当访问二维数组时（详见下文的二维数组部分）。
- 全局内存的大小和对齐要求
    
    全局内存指令支持读取或写入大小为1、2、4、8或16字节的字。当且仅当数据类型的大小为1、2、4、8或16字节且数据自然对齐（即其地址是该大小的倍数）时，对驻留在全局内存中的数据的任何访问（通过变量或指针）都会编译成单个全局内存指令。
    
    如果不满足这个大小和对齐要求，访问会编译成多条指令，并且这些指令的交错访问模式会阻止它们完全合并。因此，建议对驻留在全局内存中的数据使用满足此要求的类型。
    
    内置的向量类型自动满足对齐要求。对于结构体，可以使用编译器的对齐说明符 `__align__(8)` 或 `__align__(16)` 来强制满足大小和对齐要求，例如：
    
    ```cpp
    struct __align__(8) {
        float x;
        float y;
    };
    struct __align__(16) {
        float x;
        float y;
        float z;
    };
    
    ```
    
    驻留在全局内存中的变量的任何地址或由驱动程序或运行时API的内存分配例程返回的地址总是至少对齐到256字节。读取非自然对齐的8字节或16字节的字会产生不正确的结果（偏离几个字），因此必须特别注意保持这些类型的值或值数组的起始地址对齐。
    
    一个典型的容易忽视的情况是使用某些自定义全局内存分配方案，将多个数组的分配（多次调用 `cudaMalloc()` 或 `cuMemAlloc()`）替换为分配单个大块内存并将其划分为多个数组。在这种情况下，每个数组的起始地址会从块的起始地址偏移。
    
- 二维数组在全局内存中的访问
    
    当每个线程访问二维数组中的一个元素时，通常使用以下地址计算公式：
    
    ```cpp
    BaseAddress + width * ty + tx
    ```
    
    在此条件下，为了使这些内存访问完全合并，线程块的宽度和数组的宽度必须是warp大小的倍数。为了提高访问效率，可以将数组的宽度向上取整为warp大小的最接近倍数，并相应地填充数组的行。
    
    参考手册中描述的 `cudaMallocPitch()` 和 `cuMemAllocPitch()` 函数以及相关的内存复制函数使程序员能够编写不依赖于硬件的代码来分配符合这些约束的数组。
    
- 本地内存 Local Memory
    
    本地内存与全局内存一样，都是设备内存的一部分。本地内存被设计用于专门用于存储某些自动变量
    
    - 编译器无法确定使用常量数量索引的数组。
    - 占用过多寄存器空间的大型结构或数组。
    - 当内核使用的寄存器超过可用寄存器数量的任何变量（这也称为寄存器溢出）。
    
    通过检查PTX汇编代码（使用 `-ptx` 或 `-keep` 选项编译获得）可以判断变量是否在初始编译阶段被放置在本地内存中。此时，变量会使用 `.local` 助记符声明，并通过 `ld.local` 和 `st.local` 助记符进行访问。即使在初始阶段没有被放置在本地内存中，后续编译阶段可能仍会因其占用过多寄存器空间而决定将其放置在本地内存中。通过使用 `cuobjdump` 检查cubin对象可以判断这种情况是否发生。此外，编译器在使用 `--ptxas-options=-v` 选项编译时会报告每个内核的本地内存总使用量（lmem）。需要注意的是，一些数学函数的实现路径可能会访问本地内存。
    
    本地内存空间位于设备内存中，因此本地内存访问具有与全局内存访问相同的高延迟和低带宽，并且需要满足与内存合并相关的相同要求，如设备内存访问中所述。然而，本地内存的组织方式是连续的32位字由连续的线程ID访问。因此，只要warp中的所有线程访问相同的相对地址（例如，数组变量中的相同索引，结构变量中的相同成员），这些访问就可以完全合并。
    
    在某些计算能力为3.x的设备上，本地内存访问始终像全局内存访问一样在L1和L2中缓存（参见计算能力3.x）。
    
    在计算能力为5.x和6.x的设备上，本地内存访问始终像全局内存访问一样在L2中缓存（参见计算能力5.x和计算能力6.x）。
    
- 共享内存 Shared Memory
    
    由于共享内存在芯片上，它比本地内存或全局内存具有更高的带宽和更低的延迟。
    
    为了实现高带宽，共享内存被分成大小相等的内存模块，称为**内存银行**（banks），这些内存银行可以同时访问。因此，任何包含n个地址且这些地址位于不同内存银行的内存读或写请求可以同时被处理，从而实现比单个模块高n倍的整体带宽。
    
    如果一个内存请求中的两个地址落在相同的内存银行，就会发生**内存银行冲突**（bank conflict），这时访问必须被串行化。硬件会将发生银行冲突的内存请求拆分成若干个独立的无冲突请求，吞吐量因此下降，下降的比例等于独立请求的数量。如果独立请求的数量为n，则最初的内存请求被称为导致了n次银行冲突。
    
    为了获得最大性能，重要的是理解内存地址如何映射到内存银行，以便安排内存请求，尽量减少银行冲突。对于计算能力为3.x、5.x、6.x、7.x、8.x和9.0的设备，分别在相应的计算能力章节中描述了这一点。
    
- 常量内存 Constant Memory
    
    常量内存空间位于设备内存中，并缓存在常量缓存中。当一个请求包含多个不同的内存地址时，这个请求会被拆分成多个独立的请求，吞吐量会因此下降，下降的比例等于独立请求的数量。
    
    当发生缓存命中时，生成的请求会以常量缓存的吞吐量来处理；否则，会以设备内存的吞吐量来处理。
    
- 纹理和表面内存 Texture and Surface Memory
    
    纹理和表面内存空间位于设备内存中，并缓存在纹理缓存中。因此，纹理获取或表面读取在缓存未命中时需要一次设备内存读取，否则只需要一次纹理缓存读取。纹理缓存针对二维空间局部性进行了优化，因此同一warp的线程读取接近的二维纹理或表面地址时将实现最佳性能。此外，它设计用于流式获取，具有恒定的延迟；缓存命中减少了对DRAM带宽的需求，但不减少获取延迟。
    
    通过纹理或表面获取读取设备内存有一些优势，使其成为读取全局内存或常量内存的有利替代方案：
    
    - 如果内存读取不遵循全局内存或常量内存读取必须遵循的访问模式以获得良好性能，只要纹理获取或表面读取中存在局部性，就可以实现更高的带宽。
    - 地址计算在内核外由专用单元执行。
    - 可以在单次操作中将打包数据广播到独立的变量。
    - 8位和16位整数输入数据可以选择性地转换为范围为[0.0, 1.0]或[-1.0, 1.0]的32位浮点值（详见纹理内存）。

## 6.4 指令优化

为了最大化指令吞吐量，应用程序应：

- **最小化低吞吐量算术指令的使用**：
    - 当不影响最终结果时，可以通过降低精度来提高速度，例如使用内在函数代替常规函数（详见内在函数部分）、使用单精度浮点数代替双精度浮点数，或将非正规化的数字冲洗为零。
- **最小化由控制流指令引起的warp分歧**：
    - 详见控制流指令部分。
- **减少指令数量**：
    - 例如，通过尽可能优化同步点来减少指令数量（详见同步指令部分）或使用限制指针（详见 `__restrict__`）。

在本节中，吞吐量以每个多处理器每时钟周期的操作次数给出。对于warp大小为32，一个指令对应32个操作，因此如果N是每时钟周期的操作次数，则指令吞吐量为N/32指令每时钟周期。所有吞吐量都是针对一个多处理器的。它们必须乘以设备中的多处理器数量才能得到整个设备的吞吐量。

### 6.4.1 算数指令

给出原生指令的性能开销：

![Throughput.png](CUDA%20%E5%85%A5%E9%97%A8%20f85cae70ec504520aeb162bc10134a16/Throughput.png)

其他指令和函数是基于原生指令实现的。不同计算能力的设备可能有不同的实现，编译后的原生指令数量可能会因每个编译器版本而变化。对于复杂函数，根据输入可能存在多条代码路径。可以使用 `cuobjdump` 来检查 cubin 对象中的具体实现。一些函数的实现可以在 CUDA 头文件中找到（如 `math_functions.h`、`device_functions.h` 等）。

1. 编译优化
    1. 通常情况下，使用 `-ftz=true`（将非正规化数字冲洗为零）编译的代码性能往往高于 `-ftz=false`。类似地，使用 `-prec-div=false`（精度较低的除法）编译的代码性能往往高于 `-prec-div=true`，使用 `-prec-sqrt=false`（精度较低的平方根）编译的代码性能往往高于 `-prec-sqrt=true`。这些编译标志的详细信息请参见 `nvcc` 用户手册。
2. 单精度浮点除法
    1. `__fdividef(x, y)`（参见内在函数）提供了比除法运算符更快的单精度浮点除法。
3. 单精度浮点倒数平方根
    1. 为了保留IEEE-754语义，编译器只能在倒数和平方根都近似时（即 `-prec-div=false` 和 `-prec-sqrt=false`）优化 `1.0/sqrtf()` 为 `rsqrtf()`。因此，建议在需要时直接调用 `rsqrtf()`。
4. 单精度浮点平方根
    1. 单精度浮点平方根实现为倒数平方根，然后再求倒数，而不是倒数平方根乘以一个数，从而保证对0和无穷大给出正确的结果。
5. 正弦和余弦
    1. `sinf(x)`、`cosf(x)`、`tanf(x)`、`sincosf(x)` 以及对应的双精度指令计算成本更高，尤其是当参数 `x` 幅值较大时。参数规约代码（详见数学函数的实现）有两条路径：快速路径和慢速路径。
    2. 快速路径用于幅值较小的参数，主要由一些乘加运算组成。慢速路径用于幅值较大的参数，由实现正确结果所需的长计算组成。
    3. 当前，单精度函数的参数幅值小于 105615.0f，双精度函数的参数幅值小于 2147483648.0 时，使用快速路径。由于慢速路径需要更多寄存器，尝试通过在本地内存中存储一些中间变量来减少寄存器压力，但这可能因本地内存的高延迟和低带宽影响性能（详见设备内存访问）。当前，单精度函数使用 28 字节本地内存，双精度函数使用 44 字节，但具体数量可能会变动。
    4. 由于慢速路径的长计算和本地内存使用，当需要慢速路径规约时，这些三角函数的吞吐量比使用快速路径时低一个数量级。
6. 整数算术
    1. 整数除法和取模操作的成本较高，因为它们编译成多达 20 条指令。在某些情况下，它们可以用按位操作替代：如果 `n` 是2的幂，（i/n）等效于（i>>log2(n)），而（i%n）等效于（i&(n-1)）；如果 `n` 是字面值，编译器会执行这些转换。
    2. `__brev` 和 `__popc` 映射到单条指令，`__brevll` 和 `__popcll` 映射到几条指令。
    3. `__[u]mul24` 是过时的内在函数，不再有使用的理由。
7. 半精度算术
    1. 为了实现 16 位精度浮点加、乘或乘加的良好性能，建议使用 `half2` 数据类型进行半精度运算，并使用 `__nv_bfloat162` 进行 `__nv_bfloat16` 精度运算。然后可以使用向量内在函数（例如，`__hadd2`、`__hsub2`、`__hmul2`、`__hfma2`）在单条指令中进行两个操作。使用 `half2` 或 `__nv_bfloat162` 代替两次调用 `half` 或 `__nv_bfloat16` 可能也有助于提高其他内在函数（如 warp shuffle）的性能。
    2. 内在函数 `__halves2half2` 用于将两个半精度值转换为 `half2` 数据类型。
    3. 内在函数 `__halves2bfloat162` 用于将两个 `__nv_bfloat` 精度值转换为 `__nv_bfloat162` 数据类型。
8. 类型转换
    1. 有时，编译器必须插入转换指令，引入额外的执行周期。情况包括：
    2. 对 `char` 或 `short` 类型变量进行操作的函数，其操作数通常需要转换为 `int`。
    3. 用于单精度浮点计算的双精度浮点常量（即未定义任何类型后缀的常量，如 C/C++ 标准所规定）。
    4. 最后一种情况可以通过使用带 `f` 后缀的单精度浮点常量来避免，如 `3.141592653589793f`、`1.0f`、`0.5f`。

### 6.4.2 控制流指令

任何流控制指令（如 `if`, `switch`, `do`, `for`, `while`）都可能通过导致同一 warp 中的线程分歧（即，遵循不同的执行路径）显著影响有效指令吞吐量。如果发生这种情况，不同的执行路径必须被序列化，从而增加该 warp 执行的总指令数。

为了在控制流依赖于线程 ID 的情况下获得最佳性能，控制条件应尽量减少分歧 warp 的数量。这是可能的，因为 warp 在块中的分布是确定的，如 SIMT 架构中所述。一个简单的例子是控制条件仅依赖于 `(threadIdx / warpSize)`，其中 `warpSize` 是 warp 的大小。在这种情况下，没有 warp 会发生分歧，因为控制条件与 warp 完全对齐。

有时，编译器可能会展开循环或通过使用分支预测来优化掉短的 `if` 或 `switch` 代码块，如下所述。在这些情况下，没有 warp 会发生分歧。程序员还可以使用 `#pragma unroll` 指令控制循环展开（详见 `#pragma unroll`）。

使用分支预测时，没有任何依赖于控制条件的指令会被跳过。相反，每条指令都与一个基于控制条件设置为 true 或 false 的每线程条件码或谓词相关联，尽管每条指令都被安排执行，但只有谓词为 true 的指令才实际执行。谓词为 false 的指令不写入结果，也不评估地址或读取操作数。

### 6.4.3 同步指令

对于计算能力为 3.x 的设备，`__syncthreads()` 的吞吐量为每时钟周期 128 次操作；对于计算能力为 6.0 的设备，为每时钟周期 32 次操作；对于计算能力为 7.x 和 8.x 的设备，为每时钟周期 16 次操作；而对于计算能力为 5.x、6.1 和 6.2 的设备，为每时钟周期 64 次操作。

需要注意的是，`__syncthreads()` 可能通过迫使多处理器闲置而影响性能，详细信息见设备内存访问。

## 6.5 内存抖动

频繁分配和释放内存的应用程序可能会发现，分配调用随时间推移会变得越来越慢。这通常是由于将内存释放回操作系统供其自身使用的性质所致。为了在这方面获得最佳性能，我们建议以下几点：

- **根据实际问题大小进行内存分配**：不要试图用 `cudaMalloc`、`cudaMallocHost` 或 `cuMemCreate` 分配所有可用内存，因为这会迫使内存立即驻留，并阻止其他应用程序使用这些内存。这会给操作系统调度程序带来更大压力，或者完全阻止其他使用相同 GPU 的应用程序运行。尽早在应用程序中分配适当大小的内存，并仅在应用程序不再需要内存时进行分配。减少应用程序中 `cudaMalloc` 和 `cudaFree` 调用的次数，特别是在性能关键区域。
- **考虑使用其他内存类型**：如果应用程序无法分配足够的设备内存，考虑回退到其他内存类型，例如 `cudaMallocHost` 或 `cudaMallocManaged`，这些内存类型的性能可能不如设备内存，但能够使应用程序继续运行。
- **使用 `cudaMallocManaged` 支持超额预订**：对于支持该功能的平台，`cudaMallocManaged` 允许超额预订，并且在启用了正确的 `cudaMemAdvise` 策略后，能够使应用程序保持 `cudaMalloc` 的大部分性能。`cudaMallocManaged` 也不会强制分配驻留，直到需要或预取时才驻留，从而减轻操作系统调度程序的整体压力，更好地支持多租户使用场景。

通过遵循这些建议，可以有效减少内存抖动，提高应用程序的整体性能。

# 附录

## A 支持CUDA的device列表

## B 对C++扩展的详细描述

## C 协作组

## D CUDA动态并行

## E 虚拟内存管理

## F 流序内存分配

## G 图内存结点

## H 数学方法

## I C++语言支持

## J 纹理获取

## K CUDA计算能力

## L CUDA底层驱动API

## M CUDA环境变量

## N CUDA的统一内存
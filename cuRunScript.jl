using Test
using CUDA
using BenchmarkTools

function sequential_add!(y, x)

    val1 = size(y)

    for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

function parallel_add!(y, x)

    Threads.@threads for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

function gpu_add1!(y, x)

    for i = 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

function gpu_add2!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return
end

function gpu_add4!(y, x)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return
end

function gpu_add5!(y, x)

    val1 = size(y, 1)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    i = index
    while i <= length(y)
        @inbounds y[i] += x[i]
        i += stride
    end
    return
end

function bench_gpu1!(y, x)
    CUDA.@sync begin
        @cuda gpu_add1!(y, x)
    end
end

function bench_gpu2!(y, x)
    CUDA.@sync begin
        @cuda threads=256 gpu_add2!(y, x)
    end
end

# function bench_gpu3!(y, x)
#     numblocks = ceil(Int, length(y)/256)
#     CUDA.@sync begin
#         @cuda threads=256 blocks=numblocks gpu_add3!(y, x)
#     end
# end

function bench_gpu3!(y, x)
    kernel = @cuda launch=false gpu_add3!(y, x)
    config = launch_configuration(kernel.fun)
    threads = min(length(y), config.threads)
    blocks = cld(length(y), threads)

    CUDA.@sync begin
        kernel(y, x; threads, blocks)
    end
end

function bench_gpu4!(y, x)
    kernel = @cuda launch=false gpu_add4!(y, x)
    config = launch_configuration(kernel.fun)
    threads = min(length(y), config.threads)
    blocks = cld(length(y), threads)

    CUDA.@sync kernel(y, x; threads, blocks)
end

function bench_gpu5!(y, x)
    kernel = @cuda launch=false gpu_add5!(y, x)
    config = launch_configuration(kernel.fun)
    threads = min(length(y), config.threads)
    blocks = cld(length(y), threads)

    CUDA.@sync kernel(y, x; threads, blocks)
end

N = 2^10
# x = fill(1.0f0, N)  # a vector filled with 1.0 (Float32)
# y = fill(2.0f0, N)  # a vector filled with 2.0

# sequential_add!(y, x)
# @test all(y .== 3.0f0)

# parallel_add!(y, x)
# @test all(y .== 3.0f0)

y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0
x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0

# @cuda gpu_add1!(y_d, x_d)

# @btime bench_gpu1!($y_d, $x_d)

# bench_gpu1!(y_d, x_d)  # run it once to force compilation
# CUDA.@profile bench_gpu1!(y_d, x_d)
# CUDA.@profile bench_gpu3!(y_d, x_d)
# CUDA.@profile bench_gpu1!(y_d, x_d)
# CUDA.@profile bench_gpu4!(y_d, x_d)
# @test all(Array(y_d) .== 3.0f0)

y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0
CUDA.@profile bench_gpu5!(y_d, x_d)

# @test all(Array(y_d) .== 3.0f0)

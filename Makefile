all:
	nvcc gemm.cu -o gemm -lcublas -arch=sm_75
run:
	./gemm
clean:
	rm gemm
memcheck:
	compute-sanitizer ./gemm
profile:
	make clean
	nvcc gemm.cu -lineinfo -o gemm -lcublas -arch=sm_75
	TMPDIR=$(HOME)/.ncu_tmp ncu -f -k kernel --set full --export report ./gemm
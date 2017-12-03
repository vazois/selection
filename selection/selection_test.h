#ifndef SELECTION_TEST_H
#define SELECTION_TEST_H

template<class T,uint32_t block>
__global__ void select_and_8_for(T *gdata, uint64_t *res, uint64_t n, uint64_t match_pred){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	T a = 0;
	uint64_t match = 1;
	for(int i=0;i<8;i++){
		a = gdata[offset + i * n];
		match = match & ( a <= match_pred );
	}
	res[offset]= match;
}

template<class T,uint32_t block>
__global__ void select_and_8_for_unroll(T *gdata, uint64_t *res, uint64_t n, uint64_t match_pred){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	T a = 0;
	uint64_t match = 1;
#pragma unroll
	for(int i=0;i<8;i++){
		a = gdata[offset + i * n];
		match = match & ( a <= match_pred );
	}
	res[offset]= match;
}

template<class T,uint32_t block>
__global__ void select_and_8_register(T *gdata, uint64_t *res, uint64_t n, uint64_t match_pred){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	uint64_t match = 1;
	T a0,a1,a2,a3,a4,a5,a6,a7;
	a0 = gdata[offset]; match = match & ( a0 <= match_pred );
	a1 = gdata[offset+n]; match = match & ( a1 <= match_pred );
	a2 = gdata[offset+2*n]; match = match & ( a2 <= match_pred );
	a3 = gdata[offset+3*n]; match = match & ( a3 <= match_pred );
	a4 = gdata[offset+4*n]; match = match & ( a4 <= match_pred );
	a5 = gdata[offset+5*n]; match = match & ( a5 <= match_pred );
	a6 = gdata[offset+6*n]; match = match & ( a6 <= match_pred );
	a7 = gdata[offset+7*n]; match = match & ( a7 <= match_pred );

	res[offset]= match;
}

template<class T,uint32_t block>
__global__ void select_and_8_register_index(T *gdata, uint64_t *res, uint64_t n, uint64_t match_pred){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	uint64_t match = 1;
	T a0,a1,a2,a3,a4,a5,a6,a7;

	a0 = gdata[offset];
	a1 = gdata[offset+n];
	a2 = gdata[offset+2*n];
	a3 = gdata[offset+3*n];
	a4 = gdata[offset+4*n];
	a5 = gdata[offset+5*n];
	a6 = gdata[offset+6*n];
	a7 = gdata[offset+7*n];

	match = match & ( a0 <= match_pred );
	match = match & ( a1 <= match_pred );
	match = match & ( a2 <= match_pred );
	match = match & ( a3 <= match_pred );
	match = match & ( a4 <= match_pred );
	match = match & ( a5 <= match_pred );
	match = match & ( a6 <= match_pred );
	match = match & ( a7 <= match_pred );

	res[offset]= match;
}

#endif

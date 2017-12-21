#ifndef SELECTION_H
#define SELECTION_H

template<uint32_t block>
__global__ void init_ids(uint64_t *res, uint64_t n){
	uint64_t offset = block * blockIdx.x + threadIdx.x;
	res[offset]= offset;
}

template<class T,uint32_t block>
__global__ void select_and_for(T *gdata, uint64_t *res, uint64_t n, uint64_t d, uint64_t match_pred){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	T a = 0;
	uint64_t match = 1;
	for(int i=0;i<d;i++){
		a = gdata[offset + i * n];
		match = match & ( a > match_pred );
	}
	res[offset]= match;
}

template<class T,uint32_t block>
__global__ void select_and_for_gather(T *gdata, uint8_t *bvector, uint64_t n, uint64_t d, uint64_t match_pred){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	T a = 0;
	uint64_t match = 1;
	for(int i=0;i<d;i++){
		a = gdata[offset + i * n];
		match = match & ( a > match_pred );
	}
	bvector[offset]=match;
}

template<class T,uint32_t block>
__global__ void select_and_for_tpch(T *gdata, uint8_t *bvector, uint64_t n){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	T q = gdata[offset], d = gdata[offset + n], s = gdata[offset + 2*n];
	uint8_t match = (q < 24) & (d >= 4) & (d >= 6) & (s >= 19950101) & (s < 19960101);

	bvector[offset]=match;
}

template<class T,uint32_t block>
__global__ void select_and_for_stop(T *gdata, uint64_t *res, uint64_t n, uint64_t d, uint64_t match_pred){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	T a = 0;
	uint64_t match = 1;
	for(int i=0;i<d;i++){
		a = gdata[offset + i * n];
		match = match & ( a > match_pred );
		if(match == 0) break;
	}
	res[offset]= match;
}

template<class T,uint32_t block>
__global__ void select_and_for_return(T *gdata, uint64_t *res, uint64_t n, uint64_t d, uint64_t match_pred){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	T a = 0;
	uint64_t match = 1;
	for(int i=0;i<d;i++){
		a = gdata[offset + i * n];
		match = match & ( a > match_pred );
		if(match == 0) return;
	}
	res[offset]= match;
}

template<class T,uint32_t block>
__global__ void select_or_for(T *gdata, uint64_t *res, uint64_t n, uint64_t d, uint64_t match_pred){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	T a = 0;
	uint64_t match = 0;
	for(int i=0;i<d;i++){
		a = gdata[offset + i * n];
		match = match | ( a > match_pred );
	}
	res[offset]= match;
}

template<class T,uint32_t block>
__global__ void select_or_for_gather(T *gdata, uint8_t *bvector, uint64_t n, uint64_t d, uint64_t match_pred){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	T a = 0;
	uint64_t match = 1;
	for(int i=0;i<d;i++){
		a = gdata[offset + i * n];
		match = match | ( a > match_pred );
	}
	bvector[offset]=match;
}

template<class T,uint32_t block>
__global__ void select_or_for_stop(T *gdata, uint64_t *res, uint64_t n, uint64_t d, uint64_t match_pred){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	T a = 0;
	uint64_t match = 0;
	for(int i=0;i<d;i++){
		a = gdata[offset + i * n];
		match = match | ( a > match_pred );
		if(match == 1) break;
	}
	res[offset]= match;
}

#endif

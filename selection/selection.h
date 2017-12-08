#ifndef SELECTION_H
#define SELECTION_H


template<class T,uint32_t block>
__global__ void select_and(T *gdata, uint64_t *res, uint64_t n, uint64_t d, uint64_t match_pred){
	uint64_t offset = block * blockIdx.x + threadIdx.x;

	T a = 0;
	uint64_t match = 1;
	for(int i=0;i<d;i++){
		a = gdata[offset + i * n];
		match = match & ( a > match_pred );
	}
	res[offset]= match;
}

#endif

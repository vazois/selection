#ifndef SELECTION_PLAN_H
#define SELECTION_PLAN_H

template<class T,uint32_t block>
__global__ void select_1_and(T *c0, uint64_t n, uint32_t pred, uint8_t *res){
	uint64_t i = block * blockIdx.x + threadIdx.x;
	uint8_t check = 0x1;
	if( i < n ){
		check = (c0[i] < pred);
		if(check) res[i] = check;
	}
}

template<class T,uint32_t block>
__global__ void select_2_and(T *c0, T *c1, uint64_t n, uint32_t pred, uint8_t *res, uint32_t plan){
	uint64_t i = block * blockIdx.x + threadIdx.x;
	uint8_t check = 0x1;
	if( i < n ){
		if (plan == 0){
			check = (c0[i] < pred) & (c1[i] < pred);
		}else if(plan == 1){
			check = (c0[i] < pred) && (c1[i] < pred);
		}
		if(check) res[i] = check;
	}
}

template<class T,uint32_t block>
__global__ void select_3_and(T *c0, T *c1, T *c2, uint64_t n, uint32_t pred, uint8_t *res, uint32_t plan){
	uint64_t i = block * blockIdx.x + threadIdx.x;
	uint8_t check = 0x1;
	if( i < n ){
		if(plan == 0){
			check = (c0[i] < pred) & (c1[i] < pred) & (c2[i] < pred);
		}else if(plan == 1){
			check = ((c0[i] < pred) & (c1[i] < pred)) && (c2[i] < pred);
		}else if(plan == 2){
			check = (c0[i] < pred) && ((c1[i] < pred) & (c2[i] < pred));
		}else if(plan == 3){
			check = (c0[i] < pred) && (c1[i] < pred) && (c2[i] < pred);
		}
		if(check) res[i] = check;
	}
}

template<class T,uint32_t block>
__global__ void select_4_and(T *c0, T *c1, T *c2, T *c3, uint64_t n, uint32_t pred, uint8_t *res, uint32_t plan){
	uint64_t i = block * blockIdx.x + threadIdx.x;
	uint8_t check = 0x1;
	if( i < n ){
		if(plan == 0){
			check = ((c0[i] < pred) & (c1[i] < pred)) && (c2[i] < pred) && (c3[i] < pred);//211
		}else if(plan == 1){
			check = (c0[i] < pred) && ((c1[i] < pred) & (c2[i] < pred)) && (c3[i] < pred);//121
		}else if(plan == 2){
			check = (c0[i] < pred) & (c1[i] < pred) & (c2[i] < pred) & (c3[i] < pred);//211
		}else if(plan == 3){
			check = (c0[i] < pred) && (c1[i] < pred) && ((c2[i] < pred) & (c3[i] < pred));//112
		}else if(plan == 4){
			check = ((c0[i] < pred) & (c1[i] < pred)) && ((c2[i] < pred) & (c3[i] < pred));//22
		}else if(plan == 5){
			check = (c0[i] < pred) && ((c1[i] < pred) & (c2[i] < pred) & (c3[i] < pred));//13
		}else if(plan == 6){
			check = ((c0[i] < pred) & (c1[i] < pred) & (c2[i] < pred)) && (c3[i] < pred);//31
		}else if(plan == 7){
			check = (c0[i] < pred) && (c1[i] < pred) && (c2[i] < pred) && (c3[i] < pred);//4
		}
		if(check) res[i] = check;
	}
}

#endif

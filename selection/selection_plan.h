#ifndef SELECTION_PLAN_H
#define SELECTION_PLAN_H

#define ITEMS_PER_THREAD 16


template<class T,uint32_t bsize>
__global__ void select_1_and(T *c0, uint64_t n, uint32_t pred, uint8_t *res){
	uint64_t i = (bsize * ITEMS_PER_THREAD) * blockIdx.x + threadIdx.x;
	uint8_t check = 0x1;

	for(uint32_t stride = 0; stride <ITEMS_PER_THREAD; stride++){
		check = (c0[i] < pred);
		res[i] = check;
		i+=bsize;
	}
}

template<class T,uint32_t bsize>
__global__ void select_2_and(T *c0, T *c1, uint64_t n, uint32_t pred, uint8_t *res, uint32_t plan){
	uint64_t i = (bsize * ITEMS_PER_THREAD) * blockIdx.x + threadIdx.x;
	uint8_t check = 0x1;

	for(uint32_t stride = 0; stride <ITEMS_PER_THREAD; stride++){
		if (plan == 0){
			check = (c0[i] < pred) & (c1[i] < pred);
		}else if(plan == 1){
			check = (c0[i] < pred) && (c1[i] < pred);
		}
		res[i] = check;
		i+=bsize;
	}
}

template<class T,uint32_t bsize>
__global__ void select_3_and(T *c0, T *c1, T *c2, uint64_t n, uint32_t pred, uint8_t *res, uint32_t plan){
	uint64_t i = (bsize * ITEMS_PER_THREAD) * blockIdx.x + threadIdx.x;
	uint8_t check = 0x1;
	for(uint32_t stride = 0; stride <ITEMS_PER_THREAD; stride++){
		if (plan == 0){
			check = (c0[i] < pred) & (c1[i] < pred) & (c2[i] < pred);//3
		}else if(plan == 1){
			check = (c0[i] < pred) && (c1[i] < pred) && (c2[i] < pred);//111
		}else if(plan == 2){
			check = (c0[i] < pred) && ((c1[i] < pred) & (c2[i] < pred));//12
		}
		res[i] = check;
		i+=bsize;
	}
}

template<class T,uint32_t bsize>
__global__ void select_4_and(T *c0, T *c1, T *c2, T *c3, uint64_t n, uint32_t pred, uint8_t *res, uint32_t plan){
	uint64_t i = (bsize * ITEMS_PER_THREAD) * blockIdx.x + threadIdx.x;
	uint8_t check = 0x1;

	for(uint32_t stride = 0; stride <ITEMS_PER_THREAD; stride++){
		if(plan == 0){
			check = (c0[i] < pred) & (c1[i] < pred) & (c2[i] < pred) & (c3[i] < pred);//4
		}else if(plan == 1){
			check = (c0[i] < pred) && (c1[i] < pred) && (c2[i] < pred) && (c3[i] < pred);//1111
		}else if(plan == 2){
			check = (c0[i] < pred) && ((c1[i] < pred) & (c2[i] < pred) & (c3[i] < pred));//13
		}else if(plan == 3){
			check = (c0[i] < pred) && (c1[i] < pred) && ((c2[i] < pred) & (c3[i] < pred));//112
		}else if(plan == 4){
			check = ((c0[i] < pred) & (c1[i] < pred)) && ((c2[i] < pred) & (c3[i] < pred));//22
		}
		res[i] = check;
		i+=bsize;
	}
}

#endif

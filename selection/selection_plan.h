#ifndef SELECTION_PLAN_H
#define SELECTION_PLAN_H

#define ITEMS_PER_THREAD 16
//0 1 2 3
//1 0 0 1
//

template<class T,uint32_t block>
__global__ void init_ids2(T *res, T n){
	T offset = block * blockIdx.x + threadIdx.x;
	res[offset]= offset;
}

template<class T,uint32_t bsize>
__global__ void select_1_and(T *c0, uint64_t n, uint32_t pred, uint8_t *res){
	uint64_t i = (bsize * ITEMS_PER_THREAD) * blockIdx.x + threadIdx.x;
	uint8_t check = 0x1;

	for(uint32_t stride = 0; stride <ITEMS_PER_THREAD; stride++){
		check = (c0[i] > pred);
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
			check = (c0[i] > pred) & (c1[i] > pred);
		}else if(plan == 1){
			check = (c0[i] > pred) && (c1[i] > pred);
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
			check = (c0[i] > pred) & (c1[i] > pred) & (c2[i] > pred);//3
		}else if(plan == 1){
			check = (c0[i] > pred) && (c1[i] > pred) && (c2[i] > pred);//111
		}else if(plan == 2){
			check = (c0[i] > pred) && ((c1[i] > pred) & (c2[i] > pred));//12
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
			check = (c0[i] > pred) & (c1[i] > pred) & (c2[i] > pred) & (c3[i] > pred);//4
		}else if(plan == 1){
			check = (c0[i] > pred) && (c1[i] > pred) && (c2[i] > pred) && (c3[i] > pred);//1111
		}else if(plan == 2){
			check = (c0[i] > pred) && ((c1[i] > pred) & (c2[i] > pred) & (c3[i] < pred));//13
		}else if(plan == 3){
			check = (c0[i] > pred) && (c1[i] > pred) && ((c2[i] > pred) & (c3[i] > pred));//112
		}else if(plan == 4){
			check = ((c0[i] > pred) & (c1[i] > pred)) && ((c2[i] > pred) & (c3[i] > pred));//22
		}
		res[i] = check;
		i+=bsize;
	}
}

template<class T,uint32_t bsize>
__global__ void select_5_and(T *c0, T *c1, T *c2, T *c3, T *c4, uint64_t n, uint32_t pred, uint8_t *res, uint32_t plan){
	uint64_t i = (bsize * ITEMS_PER_THREAD) * blockIdx.x + threadIdx.x;
	uint8_t check = 0x1;

	for(uint32_t stride = 0; stride <ITEMS_PER_THREAD; stride++){
		if(plan == 0){
			check = (c0[i] > pred) & (c1[i] > pred) & (c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred);//5
		}else if(plan == 1){
			check = (c0[i] > pred) && (c1[i] > pred) && (c2[i] > pred) && (c3[i] > pred) && (c4[i] > pred);//11111
		}else if(plan == 2){
			check = (c0[i] > pred) && ((c1[i] > pred) & (c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred));//14
		}else if(plan == 3){
			check = (c0[i] > pred) && (c1[i] > pred) && ((c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred));//113
		}else if(plan == 4){
			check = ((c0[i] > pred) & (c1[i] > pred)) && ((c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred));//23
		}else if(plan == 5){
			check = (c0[i] > pred) && (c1[i] > pred) && (c2[i] > pred) && ((c3[i] > pred) & (c4[i] > pred));//1112
		}else if(plan == 6){
			check = (c0[i] > pred) && ((c1[i] > pred) & (c2[i] > pred)) && ((c3[i] > pred) & (c4[i] > pred));//122
		}
		res[i] = check;
		i+=bsize;
	}
}

template<class T,uint32_t bsize>
__global__ void select_6_and(T *c0, T *c1, T *c2, T *c3, T *c4, T *c5, uint64_t n, uint32_t pred, uint8_t *res, uint32_t plan){
	uint64_t i = (bsize * ITEMS_PER_THREAD) * blockIdx.x + threadIdx.x;
	uint8_t check = 0x1;

	for(uint32_t stride = 0; stride <ITEMS_PER_THREAD; stride++){
		if(plan == 0){
			check = (c0[i] > pred) & (c1[i] > pred) & (c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred);//6
		}else if(plan == 1){
			check = (c0[i] > pred) && (c1[i] > pred) && (c2[i] > pred) && (c3[i] > pred) && (c4[i] > pred) && (c5[i] > pred);//111111
		}else if(plan == 2){
			check = (c0[i] > pred) && ((c1[i] > pred) & (c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred));//15
		}else if(plan == 3){
			check = (c0[i] > pred) && (c1[i] > pred) && ((c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred));//114
		}else if(plan == 4){
			check = ((c0[i] > pred) & (c1[i] > pred)) && ((c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred));//24
		}else if(plan == 5){
			check = (c0[i] > pred) && (c1[i] > pred) && (c2[i] > pred) && ((c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred));//1113
		}else if(plan == 6){
			check = (c0[i] > pred) & (c1[i] > pred) & (c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred);//123
		}else if(plan == 7){
			check = ((c0[i] > pred) & (c1[i] > pred) & (c2[i] > pred)) && ((c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred));//33
		}else if(plan == 8){
			check = (c0[i] > pred) && (c1[i] > pred) && (c2[i] > pred) && (c3[i] > pred) && ((c4[i] > pred) & (c5[i] > pred));//11112
		}else if(plan == 9){
			check = (c0[i] > pred) && (c1[i] > pred) && ((c2[i] > pred) & (c3[i] > pred)) && ((c4[i] > pred) & (c5[i] > pred));//1122
		}else if(plan == 10){
			check = ((c0[i] > pred) && (c1[i] > pred)) && ((c2[i] > pred) & (c3[i] > pred)) && ((c4[i] > pred) & (c5[i] > pred));//222
		}
		res[i] = check;
		i+=bsize;
	}
}

template<class T,uint32_t bsize>
__global__ void select_7_and(T *c0, T *c1, T *c2, T *c3, T *c4, T *c5, T *c6, uint64_t n, uint32_t pred, uint8_t *res, uint32_t plan){
	uint64_t i = (bsize * ITEMS_PER_THREAD) * blockIdx.x + threadIdx.x;
	uint8_t check = 0x1;

	for(uint32_t stride = 0; stride <ITEMS_PER_THREAD; stride++){
		if(plan == 0){
			check = (c0[i] > pred) & (c1[i] > pred) & (c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred) & (c6[i] > pred);//7
		}else if(plan == 1){
			check = (c0[i] > pred) && (c1[i] > pred) && (c2[i] > pred) && (c3[i] > pred) && (c4[i] > pred) && (c5[i] > pred) && (c6[i] > pred);//1111111
		}else if(plan == 2){
			check = (c0[i] > pred) && ((c1[i] > pred) & (c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred) & (c6[i] > pred));//16
		}else if(plan == 3){
			check = (c0[i] > pred) && (c1[i] > pred) && ((c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred) & (c6[i] > pred));//115
		}else if(plan == 4){
			check = ((c0[i] > pred) & (c1[i] > pred)) && ((c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred) & (c6[i] > pred));//25
		}else if(plan == 5){
			check = (c0[i] > pred) && (c1[i] > pred) && (c2[i] > pred) && ((c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred) & (c6[i] > pred));//1114
		}else if(plan == 6){
			check = (c0[i] > pred) && ((c1[i] > pred) & (c2[i] > pred)) && ((c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred) & (c6[i] > pred));//124
		}else if(plan == 7){
			check = ((c0[i] > pred) & (c1[i] > pred)) && ((c2[i] > pred) & (c3[i] > pred)) && ((c4[i] > pred) & (c5[i] > pred) & (c6[i] > pred));//223
		}else if(plan == 8){
			check = ((c0[i] > pred) & (c1[i] > pred) & (c2[i] > pred)) && ((c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred) & (c6[i] > pred));//34
		}else if(plan == 9){
			check = (c0[i] > pred) && (c1[i] > pred) && (c2[i] > pred) && (c3[i] > pred) && (c4[i] > pred) && ((c5[i] > pred) & (c6[i] > pred));//111112
		}else if(plan == 10){
			check = (c0[i] > pred) && (c1[i] > pred) && (c2[i] > pred) && ((c3[i] > pred) & (c4[i] > pred)) && ((c5[i] > pred) & (c6[i] > pred));//11122
		}else if(plan == 11){
			check = (c0[i] > pred) && ((c1[i] > pred) & (c2[i] > pred)) && ((c3[i] > pred) & (c4[i] > pred)) && ((c5[i] > pred) & (c6[i] > pred));//1222
		}else if(plan == 12){
			check = (c0[i] > pred) && (c1[i] > pred) && ((c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred)) && ((c5[i] > pred) & (c6[i] > pred));//1132
		}else if(plan == 13){
			check = ((c0[i] > pred) & (c1[i] > pred)) && ((c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred)) && ((c5[i] > pred) & (c6[i] > pred));//232
		}
		
		res[i] = check;
		i+=bsize;
	}
}

template<class T,uint32_t bsize>
__global__ void select_8_and(T *c0, T *c1, T *c2, T *c3, T *c4, T *c5, T *c6, T *c7, uint64_t n, uint32_t pred, uint8_t *res, uint32_t plan){
	uint64_t i = (bsize * ITEMS_PER_THREAD) * blockIdx.x + threadIdx.x;
	uint8_t check = 0x1;

	for(uint32_t stride = 0; stride <ITEMS_PER_THREAD; stride++){
		if(plan == 0){
			check = (c0[i] > pred) & (c1[i] > pred) & (c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred) & (c6[i] > pred) & (c7[i] > pred);//8
		}else if(plan == 1){
			check = (c0[i] > pred) && (c1[i] > pred) && (c2[i] > pred) && (c3[i] > pred) && (c4[i] > pred) && (c5[i] > pred) && (c6[i] > pred) && (c7[i] > pred);//11111111
		}else if(plan == 2){
			check = (c0[i] > pred) && ((c1[i] > pred) & (c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred) & (c6[i] > pred) & (c7[i] > pred));//17
		}else if(plan == 3){
			check = (c0[i] > pred) && (c1[i] > pred) && ((c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred) & (c6[i] > pred) & (c7[i] > pred));//116
		}else if(plan == 4){
			check = ((c0[i] > pred) & (c1[i] > pred)) && ((c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred) & (c6[i] > pred) & (c7[i] > pred));//26
		}else if(plan == 5){
			check = (c0[i] > pred) && (c1[i] > pred) && (c2[i] > pred) && ((c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred) & (c6[i] > pred) & (c7[i] > pred));//1115
		}else if(plan == 6){
			check = (c0[i] > pred) && ((c1[i] > pred) & (c2[i] > pred)) && ((c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred) & (c6[i] > pred) & (c7[i] > pred));//125
		}else if(plan == 7){
			check = ((c0[i] > pred) & (c1[i] > pred) & (c2[i] > pred)) && ((c3[i] > pred) & (c4[i] > pred) & (c5[i] > pred) & (c6[i] > pred) & (c7[i] > pred));//35
		}else if(plan == 8){
			check = (c0[i] > pred) && (c1[i] > pred) && (c2[i] > pred) && (c3[i] > pred) && ((c4[i] > pred) & (c5[i] > pred) & (c6[i] > pred) & (c7[i] > pred));//11114
		}else if(plan == 9){
			check = (c0[i] > pred) && (c1[i] > pred) && ((c2[i] > pred) & (c3[i] > pred)) && ((c4[i] > pred) & (c5[i] > pred) & (c6[i] > pred) & (c7[i] > pred));//1124
		}else if(plan == 10){
			check = ((c0[i] > pred) & (c1[i] > pred)) && ((c2[i] > pred) & (c3[i] > pred)) && ((c4[i] > pred) & (c5[i] > pred) & (c6[i] > pred) & (c7[i] > pred));//224
		}else if(plan == 11){
			check = (c0[i] > pred) && (c1[i] > pred) && (c2[i] > pred) && (c3[i] > pred) && (c4[i] > pred) && ((c5[i] > pred) & (c6[i] > pred) & (c7[i] > pred));//111113
		}else if(plan == 12){
			check = (c0[i] > pred) && (c1[i] > pred) && (c2[i] > pred) && ((c3[i] > pred) & (c4[i] > pred)) && ((c5[i] > pred) & (c6[i] > pred) & (c7[i] > pred));//11123
		}else if(plan == 13){
			check = (c0[i] > pred) && (c1[i] > pred) && ((c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred)) && ((c5[i] > pred) & (c6[i] > pred) & (c7[i] > pred));//1133
		}else if(plan == 14){
			check = ((c0[i] > pred) & (c1[i] > pred)) && ((c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred)) && ((c5[i] > pred) & (c6[i] > pred) & (c7[i] > pred));//233
		}else if(plan == 15){
			check = (c0[i] > pred) && ((c1[i] > pred) & (c2[i] > pred) & (c3[i] > pred) & (c4[i] > pred)) && ((c5[i] > pred) & (c6[i] > pred) & (c7[i] > pred));//143
		}else if(plan == 16){
			check = (c0[i] > pred) && ((c1[i] > pred) & (c2[i] > pred)) && ((c3[i] > pred) & (c4[i] > pred)) && ((c5[i] > pred) & (c6[i] > pred) & (c7[i] > pred));//1223
		}
		res[i] = check;
		i+=bsize;
	}
}

#endif

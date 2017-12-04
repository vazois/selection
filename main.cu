#include "time/Time.h"
#include "tools/ArgParser.h"
#include "tools/File.h"
#include "time/Time.h"

#include "cuda/CudaHelper.h"

#include "selection/selection_test.h"

#define BLOCK_SIZE 512

template<class T>
void micro_bench(T *gdata, uint64_t *gres, uint64_t n, uint64_t d, uint64_t match_pred){

	//Start Processing
	dim3 grid(n/BLOCK_SIZE,1,1);
	dim3 block(BLOCK_SIZE,1,1);

	for(int i=0; i < 10; i++) select_and_for<uint64_t,BLOCK_SIZE><<<grid,block>>>(gdata,gres, n, d, match_pred);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_generic_for");
	for(int i=0; i < 10; i++) select_and_8_for<uint64_t,BLOCK_SIZE><<<grid,block>>>(gdata,gres, n, match_pred);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_for");
	for(int i=0; i < 10; i++) select_and_8_for_unroll<uint64_t,BLOCK_SIZE><<<grid,block>>>(gdata,gres, n, match_pred);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_for_unroll");
	for(int i=0; i < 10; i++) select_and_8_register<uint64_t,BLOCK_SIZE><<<grid,block>>>(gdata,gres, n, match_pred);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_register");
	for(int i=0; i < 10; i++) select_and_8_register_index<uint64_t,BLOCK_SIZE><<<grid,block>>>(gdata,gres, n, match_pred);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_register_index");

}

int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);
	uint64_t selectivity_first, selectivity_second;
	uint64_t maxValue, match_pred;

	if(!ap.exists("-f")){
		std::cout << "Missing file input!!! (-f)" << std::endl;
		exit(1);
	}

	if(!ap.exists("-mx")){
		std::cout << "Missing maximum value!!! (-mx)" << std::endl;
		exit(1);
	}

	if(ap.exists("-s")){
		selectivity_first = ap.getInt("-s");
		selectivity_second = selectivity_first;
	}else{
		selectivity_first = 10;
		selectivity_second = 100;
	}
	maxValue = ap.getInt("-mx");


	//Initialize load wrapper and pointers
	File<uint64_t> f(ap.getString("-f"),true);
	uint64_t *data = NULL;
	uint64_t *res = NULL;
	uint64_t *gdata = NULL;
	uint64_t *gres = NULL;

	cutil::safeMallocHost<uint64_t,uint64_t>(&(data),sizeof(uint64_t)*f.items()*f.rows(),"data alloc");
	cutil::safeMallocHost<uint64_t,uint64_t>(&(res),sizeof(uint64_t)*f.rows(),"res alloc");
	cutil::safeMalloc<uint64_t,uint64_t>(&(gdata),sizeof(uint64_t)*f.items()*f.rows(),"gdata alloc");
	cutil::safeMalloc<uint64_t,uint64_t>(&(gres),sizeof(uint64_t)*f.rows(),"gres alloc");

	//Load data
	f.set_transpose(true);
	f.load(data);
	//f.sample();

	//Transfer to GPU
	cutil::safeCopyToDevice<uint64_t,uint64_t>(gdata,data,sizeof(uint64_t)*f.items()*f.rows(), " copy from data to gdata ");
	std::cout << "Benchmark: (" << f.rows() << "," << f.items() << ")" <<std::endl;
	Time<msecs> t;
	for(int s = selectivity_first ; s <= selectivity_second ; s+=10){
		match_pred = maxValue/s;
		t.start();
		micro_bench(gdata,gres,f.rows(),f.items(),match_pred);
		t.lap("selectivity: "+std::to_string((float)s/maxValue));
	}


	cudaFreeHost(data);
	cudaFreeHost(res);
	cudaFree(gdata);
	cudaFree(gres);

	return 0;
}

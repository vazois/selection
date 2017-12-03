#include "time/Time.h"
#include "tools/ArgParser.h"
#include "tools/File.h"

#include "cuda/CudaHelper.h"

#include "selection/selection_test.h"

#define BLOCK_SIZE 512



int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);
	if(!ap.exists("-f")){
		std::cout << "Missing file input!!!" << std::endl;
		exit(1);
	}

	//Initialize load wrapper and pointers
	File<uint64_t> f(ap.getString("-f"),true);
	uint64_t *data = NULL;
	uint64_t *res = NULL;
	uint64_t *gdata = NULL;
	uint64_t *gres = NULL;
	uint64_t match_pred = f.rows()/10;

	cutil::safeMallocHost<uint64_t,uint64_t>(&(data),sizeof(uint64_t)*f.items()*f.rows(),"data alloc");
	cutil::safeMallocHost<uint64_t,uint64_t>(&(res),sizeof(uint64_t)*f.rows(),"res alloc");
	cutil::safeMalloc<uint64_t,uint64_t>(&(gdata),sizeof(uint64_t)*f.items()*f.rows(),"gdata alloc");
	cutil::safeMalloc<uint64_t,uint64_t>(&(gres),sizeof(uint64_t)*f.rows(),"gres alloc");

	//Load data
	f.set_transpose(true);
	f.load(data);
	f.sample();

	//Transfer to GPU
	cutil::safeCopyToDevice<uint64_t,uint64_t>(gdata,data,sizeof(uint64_t)*f.items()*f.rows(), " copy from data to gdata ");

	//Start Processing
	dim3 grid(f.rows()/BLOCK_SIZE,1,1);
	dim3 block(BLOCK_SIZE,1,1);

	for(int i=0; i < 10; i++) select_and_8_for<uint64_t,BLOCK_SIZE><<<grid,block>>>(gdata,gres, f.rows(),match_pred);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_for");
	for(int i=0; i < 10; i++) select_and_8_for_unroll<uint64_t,BLOCK_SIZE><<<grid,block>>>(gdata,gres, f.rows(),match_pred);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_for_unroll");
	for(int i=0; i < 10; i++) select_and_8_register<uint64_t,BLOCK_SIZE><<<grid,block>>>(gdata,gres, f.rows(),match_pred);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_register");
	for(int i=0; i < 10; i++) select_and_8_register_index<uint64_t,BLOCK_SIZE><<<grid,block>>>(gdata,gres, f.rows(),match_pred);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_register_index");

	cudaFreeHost(data);
	cudaFreeHost(res);
	cudaFree(gdata);
	cudaFree(gres);

	return 0;
}

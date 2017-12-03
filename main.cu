#include "time/Time.h"
#include "tools/ArgParser.h"
#include "tools/File.h"

#include "cuda/CudaHelper.h"


int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);
	if(!ap.exists("-f")){
		std::cout << "Missing file input!!!" << std::endl;
		exit(1);
	}

	File<uint64_t> f(ap.getString("-f"),true);
	uint64_t *data = NULL;
	cutil::safeMallocHost<uint64_t,uint64_t>(&(data),sizeof(uint64_t)*f.items()*f.rows(),"data alloc");

	f.load(data);
	f.sample();

	//uint64_t data = NULL;
	//cutil::safeMallocHost<uint64_t,uint64_t>(&(data),sizeof(uint64_t)*this->n,"ctupples alloc");

	cudaFreeHost(data);

	return 0;
}

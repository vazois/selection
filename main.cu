#include <cub/cub.cuh>
#include "time/Time.h"
#include "tools/ArgParser.h"
#include "tools/File.h"
#include "time/Time.h"

#include "cuda/CudaHelper.h"

#include "selection/selection_test.h"
#include "selection/selection.h"
#include "selection/selection_plan.h"

#include <cstdlib>
#include <ctime>
#include <cmath>

#define BLOCK_SIZE 512
#define MAX_PRED 100
#define MIN_PRED 1

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

template<class T>
void micro_bench2(T *gdata, uint64_t *gres, uint64_t n, uint64_t d, uint64_t match_pred, uint64_t iter, uint64_t and_){
	//Start Processing
	dim3 grid(n/BLOCK_SIZE,1,1);
	dim3 block(BLOCK_SIZE,1,1);

	if (and_ == 0){
//		std::cout << "items:" << d << std::endl;
		for(uint64_t i=0; i < iter; i++) select_and_for<uint64_t,BLOCK_SIZE><<<grid,block>>>(gdata,gres, n, d, match_pred);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_generic_for");
	}else{
		for(uint64_t i=0; i < iter; i++) select_or_for_stop<uint64_t,BLOCK_SIZE><<<grid,block>>>(gdata,gres, n, d, match_pred);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_generic_for");
	}

}

template<class T>
void micro_bench3(T *gdata, uint64_t *gres, uint64_t *gres_out, uint8_t *bvector, uint64_t *dnum,void *d_temp_storage,size_t temp_storage_bytes, uint64_t n, uint64_t d, uint64_t match_pred, uint64_t iter, uint64_t and_){
	//Start Processing
	dim3 grid(n/BLOCK_SIZE,1,1);
	dim3 block(BLOCK_SIZE,1,1);

	if (and_ == 0){
		//for(uint64_t i=0; i < iter; i++) select_and_for_stop<uint64_t,BLOCK_SIZE><<<grid,block>>>(gdata,gres, n, d, match_pred);
		for(uint64_t i=0; i < iter; i++){
			select_and_for_gather<uint64_t,BLOCK_SIZE><<<grid,block>>>(gdata,bvector, n, d, match_pred);
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_generic_for_gather 2");
			cub::DevicePartition::Flagged(d_temp_storage,temp_storage_bytes,gres,bvector,gres_out,dnum, n);
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_generic_for");
		}
	}else{
		for(uint64_t i=0; i < iter; i++){
			select_or_for_gather<uint64_t,BLOCK_SIZE><<<grid,block>>>(gdata,bvector, n, d, match_pred);
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_generic_for_gather 2");
			cub::DevicePartition::Flagged(d_temp_storage,temp_storage_bytes,gres,bvector,gres_out,dnum, n);
			cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_generic_for");
		}
	}
}

template<class T>
void micro_bench4(T *gdata, uint64_t *gres, uint64_t *gres_out, uint8_t *bvector, uint64_t *dnum,void *d_temp_storage,size_t temp_storage_bytes, uint64_t n, uint64_t d, uint64_t match_pred, uint64_t iter, uint64_t and_){
	//Start Processing
	dim3 grid(n/BLOCK_SIZE,1,1);
	dim3 block(BLOCK_SIZE,1,1);

	for(uint64_t i=0; i < iter; i++){
		select_and_for_tpch<uint64_t,BLOCK_SIZE><<<grid,block>>>(gdata,bvector, n);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_generic_for_gather 2");
		cub::DevicePartition::Flagged(d_temp_storage,temp_storage_bytes,gres,bvector,gres_out,dnum, n);
		cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_generic_for");
	}
}

int fetch(uint64_t &d,uint64_t &mx, float &s, FILE *f){
	int dd,mxx;
	float ss;
	//fscanf(f,"%i",&dd);
	int r = fscanf(f,"%i,%f,%i",&dd,&ss,&mxx);
	d=dd;
	mx =mxx;
	s =ss;
	//std::cout << "<<" <<dd << "," << mxx << "," << ss << std::endl;
	return r;
}

template<class T>
void init_relation(T *data, uint64_t n, uint64_t d){
	srand(time(NULL));
	for(uint64_t i = 0; i < n; i++){
		for(uint64_t m = 0; m < d; m++){
			data[m*n + i] = MIN_PRED + (rand() % static_cast<T>(MAX_PRED - MIN_PRED + 1));
		}
	}
}

template<class T>
void micro_bench5(uint64_t n, uint64_t d){
	T *data = (T *)malloc(sizeof(T)*n*d);
	T *gdata;
	uint8_t *res,*hres;

	T *d_in = NULL;
	T *d_out = NULL;
	T *d_num_selected_out = NULL;
	void *d_temp_storage = NULL;
	size_t temp_storage_bytes;

	init_relation<uint32_t>(data,n,d);

	hres = (uint8_t *)malloc(sizeof(uint8_t)*n);
	cutil::safeMalloc<T,uint64_t>(&(gdata),sizeof(T)*n*d,"gdata alloc");//data in GPU
	cutil::safeMalloc<uint8_t,uint64_t>(&(res),sizeof(uint8_t)*n*d,"gdata alloc");//data in GPU
	cutil::safeCopyToDevice<T,uint64_t>(gdata,data,sizeof(T)*n*d, " copy from data to gdata ");

	cutil::safeMalloc<uint32_t,uint64_t>(&(d_in),sizeof(uint32_t)*n,"d_in alloc");//
	cutil::safeMalloc<uint32_t,uint64_t>(&(d_out),sizeof(uint32_t)*n,"d_out alloc");//
	cutil::safeMalloc<uint32_t,uint64_t>(&(d_num_selected_out),sizeof(uint32_t),"d_num_selected_out alloc");
	cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, res, d_out, d_num_selected_out, n);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	dim3 grid2(n/BLOCK_SIZE,1,1);
	dim3 block2(BLOCK_SIZE,1,1);
	init_ids2<T,BLOCK_SIZE><<<grid2,block2>>>(d_in,n);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init ids");//synchronize

	dim3 block(BLOCK_SIZE,1,1);
	dim3 grid((n-1)/(BLOCK_SIZE*ITEMS_PER_THREAD),1,1);
	Time<msecs> t;
	double elapsedTime;
	for(uint32_t p = 1; p <= d; p++){
		std::cout << "predicates: " << p << std::endl;

		for(int32_t s = 10; s >= 0; s--){
			double prob = pow(((double)s)/10,(1.0/(double)p));
			T pred = (T)((double)MAX_PRED -(((double)MAX_PRED)*prob));
			//std::cout << p << "," << (((double)s)/10) << "," << pred << std::endl;

			T *c0 = &gdata[0];
			T *c1 = &gdata[n];
			T *c2 = &gdata[2*n];
			T *c3 = &gdata[3*n];
			T *c4 = &gdata[4*n];
		//	T *c5 = &gdata[5*n];
		//	T *c6 = &gdata[6*n];
		//	T *c7 = &gdata[7*n];
			std::cout << std::fixed << std::setprecision(8);

			if(p == 1){
				t.start();
				select_1_and<T,BLOCK_SIZE><<<grid,block>>>(&gdata[0],n,pred,res);
				cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_1");//synchronize
				cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, res, d_out, d_num_selected_out, n);
				cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing DeviceSelect");//synchronize
				elapsedTime=t.lap();
				std::cout << elapsedTime << std::endl;
			}else if(p == 2){
				//std::cout <<"(" << (((double)s)/10)*100 << "," << pred <<"):";
				for(uint32_t pp = 0; pp < 2;pp++){
					t.start();
					select_2_and<T,BLOCK_SIZE><<<grid,block>>>(&gdata[0],&gdata[n],n,pred,res,pp);
					cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_2");//synchronize
					cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, res, d_out, d_num_selected_out, n);
					cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing DeviceSelect");//synchronize
					elapsedTime=t.lap();
					std::cout << elapsedTime << " ";
				}
				cutil::safeCopyToHost<uint8_t,uint64_t>(hres,res,sizeof(uint8_t)*n, " copy from res to hres ");
				uint32_t count = 0;
				for(uint32_t i = 0; i < n; i++) if(hres[i] == 1) count++;
//				std::cout <<"s: " <<count << "," << ((double)count) / ((double)n) << " --- " << ((double)(10 - s))/10<< std::endl;
				std::cout << ((double)count) / ((double)n);
				std::cout << std::endl;
			}else if(p == 3){
				t.start();
				select_3_and<T,BLOCK_SIZE><<<grid,block>>>(c0,c1,c2,n,pred,res,0);
				cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_3");//synchronize
				cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, res, d_out, d_num_selected_out, n);
				cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing DeviceSelect");//synchronize
				elapsedTime=t.lap();
				std::cout << elapsedTime << " ";

				t.start();
				select_3_and<T,BLOCK_SIZE><<<grid,block>>>(c0,c1,c2,n,pred,res,1);
				cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_3");//synchronize
				cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, res, d_out, d_num_selected_out, n);
				cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing DeviceSelect");//synchronize
				elapsedTime=t.lap();
				std::cout << elapsedTime << " ";

				for(uint32_t pp = 2; pp < 3;pp++){
					t.start();
					select_3_and<T,BLOCK_SIZE><<<grid,block>>>(c0,c1,c2,n,pred,res,pp);
					cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_3");//synchronize
					cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, res, d_out, d_num_selected_out, n);
					cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing DeviceSelect");//synchronize
					elapsedTime=t.lap();
					std::cout << elapsedTime << " ";

					t.start();
					select_3_and<T,BLOCK_SIZE><<<grid,block>>>(c1,c0,c2,n,pred,res,pp);
					cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_3");//synchronize
					cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, res, d_out, d_num_selected_out, n);
					cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing DeviceSelect");//synchronize
					elapsedTime=t.lap();
					std::cout << elapsedTime << " ";

					t.start();
					select_3_and<T,BLOCK_SIZE><<<grid,block>>>(c2,c0,c1,n,pred,res,pp);
					cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_3");//synchronize
					cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, res, d_out, d_num_selected_out, n);
					cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing DeviceSelect");//synchronize
					elapsedTime=t.lap();
					std::cout << elapsedTime << " ";
				}
				cutil::safeCopyToHost<uint8_t,uint64_t>(hres,res,sizeof(uint8_t)*n, " copy from res to hres ");
				uint32_t count = 0;
				for(uint32_t i = 0; i < n; i++) if(hres[i] == 1) count++;
//				std::cout <<"s: " <<count << "," << ((double)count) / ((double)n) << " --- " << ((double)(10 - s))/10<< std::endl;
				std::cout << (((double)count) / ((double)n))*100;
				std::cout << std::endl;
			}else if(p == 4){
				t.start();
				select_4_and<T,BLOCK_SIZE><<<grid,block>>>(c0,c1,c2,c3,n,pred,res,0);
				cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_4");//synchronize
				cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, res, d_out, d_num_selected_out, n);
				cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing DeviceSelect");//synchronize
				elapsedTime=t.lap();
				std::cout << elapsedTime << " ";

				t.start();
				select_4_and<T,BLOCK_SIZE><<<grid,block>>>(c0,c1,c2,c3,n,pred,res,1);
				cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_4");//synchronize
				cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, res, d_out, d_num_selected_out, n);
				cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing DeviceSelect");//synchronize
				elapsedTime=t.lap();
				std::cout << elapsedTime << " ";

			    std::string s = "0123";
			    for(uint32_t pp = 2; pp < 5;pp++){
			    	double minElapsedTime = 1024*1024*1024;
			    	do {
			    		//std::cout << s << '\n';
			    		int i0 = s[0] - '0';
			    		int i1 = s[1] - '0';
			    		int i2 = s[2] - '0';
			    		int i3 = s[3] - '0';
			    		//std::cout << i0 << "," << i1 << "," << i2 << ","<< i3 << std::endl;

			    		c0 = &gdata[i0*n];
			    		c1 = &gdata[i1*n];
			    		c2 = &gdata[i2*n];
			    		c3 = &gdata[i3*n];

						t.start();
						select_4_and<T,BLOCK_SIZE><<<grid,block>>>(c0,c1,c2,c3,n,pred,res,pp);
						cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_4");//synchronize
						cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, res, d_out, d_num_selected_out, n);
						cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing DeviceSelect");//synchronize
						elapsedTime=t.lap();
						minElapsedTime = std::min(elapsedTime,minElapsedTime);
			    	}while(std::next_permutation(s.begin(), s.end()));
					std::cout << minElapsedTime << " ";
			    }
				cutil::safeCopyToHost<uint8_t,uint64_t>(hres,res,sizeof(uint8_t)*n, " copy from res to hres ");
				uint32_t count = 0;
				for(uint32_t i = 0; i < n; i++) if(hres[i] == 1) count++;
//				std::cout <<"s: " <<count << "," << ((double)count) / ((double)n) << " --- " << ((double)(10 - s))/10<< std::endl;
				std::cout << (((double)count) / ((double)n))*100;
				std::cout << std::endl;
			}else if(p == 5){
				t.start();
				select_5_and<T,BLOCK_SIZE><<<grid,block>>>(c0,c1,c2,c3,c4,n,pred,res,0);
				cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_5");//synchronize
				cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, res, d_out, d_num_selected_out, n);
				cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing DeviceSelect");//synchronize
				elapsedTime=t.lap();
				std::cout << elapsedTime << " ";

				t.start();
				select_5_and<T,BLOCK_SIZE><<<grid,block>>>(c0,c1,c2,c3,c4,n,pred,res,1);
				cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_5");//synchronize
				cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, res, d_out, d_num_selected_out, n);
				cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing DeviceSelect");//synchronize
				elapsedTime=t.lap();
				std::cout << elapsedTime << " ";

			    std::string s = "01234";
			    for(uint32_t pp = 2; pp < 7;pp++){
			    	double minElapsedTime = 1024*1024*1024;
			    	do {
			    		//std::cout << s << '\n';
			    		int i0 = s[0] - '0';
			    		int i1 = s[1] - '0';
			    		int i2 = s[2] - '0';
			    		int i3 = s[3] - '0';
			    		int i4 = s[4] - '0';
			    		//std::cout << i0 << "," << i1 << "," << i2 << ","<< i3 << std::endl;

			    		c0 = &gdata[i0*n];
			    		c1 = &gdata[i1*n];
			    		c2 = &gdata[i2*n];
			    		c3 = &gdata[i3*n];
			    		c4 = &gdata[i4*n];

						t.start();
						select_5_and<T,BLOCK_SIZE><<<grid,block>>>(c0,c1,c2,c3,c4,n,pred,res,pp);
						cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing select_5");//synchronize
						cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, res, d_out, d_num_selected_out, n);
						cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing DeviceSelect");//synchronize
						elapsedTime=t.lap();
						minElapsedTime = std::min(elapsedTime,minElapsedTime);
			    	}while(std::next_permutation(s.begin(), s.end()));
					std::cout << minElapsedTime << " ";
			    }
				cutil::safeCopyToHost<uint8_t,uint64_t>(hres,res,sizeof(uint8_t)*n, " copy from res to hres ");
				uint32_t count = 0;
				for(uint32_t i = 0; i < n; i++) if(hres[i] == 1) count++;
//				std::cout <<"s: " <<count << "," << ((double)count) / ((double)n) << " --- " << ((double)(10 - s))/10<< std::endl;
				std::cout << (((double)count) / ((double)n))*100;
				std::cout << std::endl;
			}
		}
	}

	free(data);
	free(hres);
	cudaFree(gdata);
	cudaFree(res);

	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(d_num_selected_out);
	cudaFree(d_temp_storage);
}

int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);

	if(!ap.exists("-n")){
		std::cout << "Missing cardinality input!!! (-n)" << std::endl;
		exit(1);
	}

	if(!ap.exists("-d")){
		std::cout << "Missing dimensionality!!! (-d)" << std::endl;
		exit(1);
	}
	uint64_t n = ap.getInt("-n");
	uint64_t d = ap.getInt("-d");
	std::cout << "N:" << n << "," << "D:" << d << std::endl;

	cudaSetDevice(1);
	micro_bench5<uint32_t>(n,d);

}

int main2(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);

	if(!ap.exists("-f")){
		std::cout << "Missing file input!!! (-f)" << std::endl;
		exit(1);
	}

	if(!ap.exists("-t")){
		std::cout << "Missing query type!!! (-t)" << std::endl;
		exit(1);
	}


	uint64_t mx=0,d=0;
	float s=0;
	uint64_t and_ = ap.getInt("-t");

	//Initialize load wrapper and pointers
	File<uint64_t> f(ap.getString("-f"),true);
	uint64_t *data = NULL;
	uint64_t *gdata = NULL;
	uint8_t *bvector = NULL;
	uint64_t *gres = NULL;
	uint64_t *gres_out = NULL;
	uint64_t *dnum = NULL;

	void *d_temp_storage = NULL;
	size_t temp_storage_bytes;

	cutil::safeMallocHost<uint64_t,uint64_t>(&(data),sizeof(uint64_t)*f.items()*f.rows(),"data alloc");//data from file
	cutil::safeMalloc<uint64_t,uint64_t>(&(gdata),sizeof(uint64_t)*f.items()*f.rows(),"gdata alloc");//data in GPU
	cutil::safeMalloc<uint64_t,uint64_t>(&(gres),sizeof(uint64_t)*f.rows(),"gres alloc");//row ids
	cutil::safeMalloc<uint8_t,uint64_t>(&(bvector),sizeof(uint8_t)*f.rows(),"bvector alloc");//boolean vector for evaluated rows
	cutil::safeMalloc<uint64_t,uint64_t>(&(gres_out),sizeof(uint64_t)*f.rows(),"gres_out alloc");//qualifying rows
	cutil::safeMalloc<uint64_t,uint64_t>(&(dnum),sizeof(uint64_t),"dnum alloc");//number of rows qualified

	cub::DevicePartition::Flagged(d_temp_storage,temp_storage_bytes,gres,bvector,gres_out,dnum, f.rows());//call to allocate temp_storage
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing ids partition");//synchronize
	cutil::safeMalloc<void,uint64_t>(&(d_temp_storage),temp_storage_bytes,"tmp_storage alloc");//alloc temp storage

	dim3 grid(f.rows()/BLOCK_SIZE,1,1);
	dim3 block(BLOCK_SIZE,1,1);

	init_ids<BLOCK_SIZE><<<grid,block>>>(gres,f.rows());
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init ids");//synchronize

	//Load data
	f.set_transpose(true);
	f.load(data);
	//f.sample();

	//Transfer to GPU
	cutil::safeCopyToDevice<uint64_t,uint64_t>(gdata,data,sizeof(uint64_t)*f.items()*f.rows(), " copy from data to gdata ");

	FILE *fa;
	fa = fopen("args.out", "r");
	uint64_t iter = 10;

	if(and_ < 2){
		while (fetch(d,mx,s,fa) >0){
			Time<msecs> t;
			t.start();
			micro_bench3<uint64_t>(gdata,gres,gres_out,bvector,dnum,d_temp_storage,temp_storage_bytes,f.rows(),d,mx,iter,and_);
			std::cout << s << "," << d << "," << t.lap()/iter <<std::endl;
			if(s == 1) std::cout << std::endl;
		}
	}else{
		Time<msecs> t;
		t.start();
		micro_bench4<uint64_t>(gdata,gres,gres_out,bvector,dnum,d_temp_storage,temp_storage_bytes,f.rows(),d,mx,iter,and_);
		std::cout << s << "," << d << "," << t.lap()/iter <<std::endl;
	}

	fclose(fa);

	cudaFreeHost(data);
	cudaFree(gdata);
	cudaFree(gres);
	cudaFree(bvector);
	cudaFree(gres_out);
	cudaFree(d_temp_storage);
	cudaFree(dnum);

	return 0;
}

int main3(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);

	if(!ap.exists("-f")){
		std::cout << "Missing file input!!! (-f)" << std::endl;
		exit(1);
	}

	if(!ap.exists("-mx")){
		std::cout << "Missing maximum value!!! (-mx)" << std::endl;
		exit(1);
	}

	if(!ap.exists("-s")){
		std::cout << "Missing selectivity!!! (-s)" << std::endl;
		exit(1);
	}

	if(!ap.exists("-d")){
		std::cout << "Missing predicate size!!! (-d)" << std::endl;
		exit(1);
	}

	if(!ap.exists("-t")){
		std::cout << "Missing query type!!! (-t)" << std::endl;
		exit(1);
	}

	uint64_t mx = ap.getInt("-mx");
	float s = ap.getFloat("-s");
	uint64_t d = ap.getInt("-d");
	uint64_t and_ = ap.getInt("-t");

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

	uint64_t iter = 10;
	Time<msecs> t;
	t.start();
	micro_bench2(gdata,gres, f.rows(), d, mx, iter, and_);
	//std::cout << "<" << mx << "," << s << "," << d << "> : " << t.lap() <<std::endl;
	//std::cout << "selectivity: " << s << " pred: " << d << " time(ms): " << t.lap()/iter <<std::endl;
	std::cout << s << "," << d << "," << t.lap()/iter <<std::endl;

	//micro_bench2(gdata,gres, n, d, match_pred);

	cudaFreeHost(data);
	cudaFreeHost(res);
	cudaFree(gdata);
	cudaFree(gres);

	return 0;
}

int main4(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);

	if(!ap.exists("-f")){
		std::cout << "Missing file input!!! (-f)" << std::endl;
		exit(1);
	}

	if(!ap.exists("-mx")){
		std::cout << "Missing maximum value!!! (-mx)" << std::endl;
		exit(1);
	}

	if(!ap.exists("-s")){
		std::cout << "Missing selectivity!!! (-s)" << std::endl;
		exit(1);
	}

	if(!ap.exists("-d")){
		std::cout << "Missing predicate size!!! (-d)" << std::endl;
		exit(1);
	}

	if(!ap.exists("-t")){
		std::cout << "Missing query type!!! (-t)" << std::endl;
		exit(1);
	}

	uint64_t mx = ap.getInt("-mx");
	float s = ap.getFloat("-s");
	uint64_t d = ap.getInt("-d");
	uint64_t and_ = ap.getInt("-t");

	//Initialize load wrapper and pointers
	File<uint64_t> f(ap.getString("-f"),true);
	uint64_t *data = NULL;
	uint64_t *gdata = NULL;
	uint8_t *bvector = NULL;
	uint64_t *gres = NULL;
	uint64_t *gres_out = NULL;
	uint64_t *dnum = NULL;

	void *d_temp_storage = NULL;
	size_t temp_storage_bytes;

	cutil::safeMallocHost<uint64_t,uint64_t>(&(data),sizeof(uint64_t)*f.items()*f.rows(),"data alloc");//data from file
	cutil::safeMalloc<uint64_t,uint64_t>(&(gdata),sizeof(uint64_t)*f.items()*f.rows(),"gdata alloc");//data in GPU
	cutil::safeMalloc<uint64_t,uint64_t>(&(gres),sizeof(uint64_t)*f.rows(),"gres alloc");//row ids
	cutil::safeMalloc<uint8_t,uint64_t>(&(bvector),sizeof(uint8_t)*f.rows(),"bvector alloc");//boolean vector for evaluated rows
	cutil::safeMalloc<uint64_t,uint64_t>(&(gres_out),sizeof(uint64_t)*f.rows(),"gres_out alloc");//qualifying rows
	cutil::safeMalloc<uint64_t,uint64_t>(&(dnum),sizeof(uint64_t),"dnum alloc");//number of rows qualified

	cub::DevicePartition::Flagged(d_temp_storage,temp_storage_bytes,gres,bvector,gres_out,dnum, f.rows());//call to allocate temp_storage
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing ids partition");//synchronize
	cutil::safeMalloc<void,uint64_t>(&(d_temp_storage),temp_storage_bytes,"tmp_storage alloc");//alloc temp storage

	dim3 grid(f.rows()/BLOCK_SIZE,1,1);
	dim3 block(BLOCK_SIZE,1,1);

	init_ids<BLOCK_SIZE><<<grid,block>>>(gres,f.rows());
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing init ids");//synchronize

	//Load data
	f.set_transpose(true);
	f.load(data);
	//f.sample();

	//Transfer to GPU
	cutil::safeCopyToDevice<uint64_t,uint64_t>(gdata,data,sizeof(uint64_t)*f.items()*f.rows(), " copy from data to gdata ");

	uint64_t iter = 10;
	Time<msecs> t;
	t.start();
	micro_bench3<uint64_t>(gdata,gres,gres_out,bvector,dnum,d_temp_storage,temp_storage_bytes,f.rows(),d,mx,iter,and_);
	std::cout << s << "," << d << "," << t.lap()/iter <<std::endl;

//	t.start();
//	micro_bench2(gdata,gres, f.rows(), d, mx, iter, and_);
//	std::cout << s << "," << d << "," << t.lap()/iter <<std::endl;

	cudaFreeHost(data);
	cudaFree(gdata);
	cudaFree(gres);
	cudaFree(bvector);
	cudaFree(gres_out);
	cudaFree(d_temp_storage);
	cudaFree(dnum);

	return 0;
}

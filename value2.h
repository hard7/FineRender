#ifndef __CORE_DISTR_VAR_H__
#define __CORE_DISTR_VAR_H__


#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <helper/helper_cuda.h>
#include <helper/helper_math.h>
#include <cassert>
#include <cmath>
#include "frame.h"

using std::cout;
using std::endl;

enum used_host_e {NOT_INIT_HOST_MEMORY, INIT_BASE_HOST_MEMORY, INIT_ALL_HOST_MEMORY};

//FIXME using namespace. using more concrete class name
template<class T> //FIXME for primiteve types use typename
struct value_t {
	std::vector<T*> gpu_pointers;       //FIXME incapsulate fields
	T* host_pointer;                    //FIXME divide responsibility
	const frame_t& frame;
	int state_count;
	int items_per_cell;
	used_host_e used_host;              //FIXME don't make premature optimization (by memory too)

	//FIXME complicate to initialize by constructor. may be to use builder
	value_t(const frame_t& frame_, int state_count_, int items_per_cell_, used_host_e used_host_)
	: frame(frame_), state_count(state_count_), items_per_cell(items_per_cell_), host_pointer(nullptr), used_host(used_host_) {
		gpu_pointers.resize(state_count, nullptr);
		long byte_size = frame.get_item_count() * items_per_cell * sizeof(T);
				
		checkCudaErrors( cudaSetDevice(frame.dev_id) );
		
		if(used_host == INIT_BASE_HOST_MEMORY) {
			checkCudaErrors( cudaMallocHost((void **) &host_pointer, frame.items_for_calculate * items_per_cell * sizeof(T)) );
		}
		else if(used_host == INIT_ALL_HOST_MEMORY) {
			checkCudaErrors( cudaMallocHost((void **) &host_pointer, frame.get_item_count() * items_per_cell * sizeof(T)) );
		}
		
		for(auto& gpu_ptr : gpu_pointers) {
			checkCudaErrors( cudaMalloc((void **) &gpu_ptr, byte_size) );
		}
	}
	
	long get_bytes_of_real_part() const {
		return frame.items_for_calculate * sizeof(T);
	}
	
	void copyHtoD() {
		copyHtoD(state_count - 1);
	}
	
	void copyDtoH() {
		copyDtoH(state_count - 1);
	}

	void copyHtoD(int state) {
		assert(used_host == INIT_BASE_HOST_MEMORY or used_host == INIT_ALL_HOST_MEMORY);
		int offset = (used_host == INIT_BASE_HOST_MEMORY ? frame.left : 0);
		int byte_count = sizeof(T) * items_per_cell * (used_host == INIT_BASE_HOST_MEMORY ? frame.items_for_calculate : frame.get_item_count());
		
		checkCudaErrors( cudaSetDevice(frame.dev_id) );
		checkCudaErrors( cudaMemcpy(gpu_pointers[state] + offset, host_pointer, byte_count, cudaMemcpyHostToDevice) );
	}
	
	void copyDtoH(int state) {
		assert(used_host == INIT_BASE_HOST_MEMORY or used_host == INIT_ALL_HOST_MEMORY);
		int offset = (used_host == INIT_BASE_HOST_MEMORY ? frame.left : 0);
		int byte_count = sizeof(T) * items_per_cell * (used_host == INIT_BASE_HOST_MEMORY ? frame.items_for_calculate : frame.get_item_count());
		
		checkCudaErrors( cudaSetDevice(frame.dev_id) );
		checkCudaErrors( cudaMemcpy(host_pointer, gpu_pointers[state] + offset, byte_count, cudaMemcpyDeviceToHost) );
	}
	
	void destroy() {
		checkCudaErrors( cudaSetDevice(frame.dev_id) );
		
		if(host_pointer) {
			checkCudaErrors( cudaFreeHost(host_pointer) );
			host_pointer = nullptr;
		}
		
		for(auto& gpu_ptr : gpu_pointers) {
			checkCudaErrors( cudaFree(gpu_ptr) );
			gpu_ptr = nullptr;
		}
	}
	
	T* get_host_pointer() {
		return host_pointer;
	}
	
	T* left_dst(int state, int n = 0) {
		assert(frame.has_left_neighbor);
		return gpu_pointers.at(state) + frame.offset_to_left_dst(n);
	}
	
	const T* left_src(int state, int n = 0) const {
		assert(frame.has_left_neighbor);
		return gpu_pointers.at(state) + frame.offset_to_left_src(n);
	}
	
	T* right_dst(int state, int n = 0) {
		assert(frame.has_right_neighbor);
		return gpu_pointers.at(state) + frame.offset_to_right_dst(n);
	}
	
	const T* right_src(int state, int n = 0) const {
		assert(frame.has_right_neighbor);
		return gpu_pointers.at(state) + frame.offset_to_right_src(n);
	}
	
	int get_dev_id() const {
		return frame.dev_id;
	}
	
	const frame_t& get_frame() const {
		return frame;
	}
	
	T* operator[](int id) {
		return gpu_pointers[id] + frame.left;
	}
	
	const T* operator[](int id) const {
		return gpu_pointers[id] + frame.left;
	}
	
	void swap() {
		gpu_pointers.insert(gpu_pointers.end(), *gpu_pointers.begin());
		gpu_pointers.erase(gpu_pointers.begin());
	}
};


template<class T>
class distributed_value_t : public std::vector< value_t<T> > {
	const distributed_frame_t& distributed_frame;
	int state_count;
	int items_per_cell;
	
	
public:
	distributed_value_t(const distributed_frame_t& distributed_frame_, int state_count_, int items_per_cell_=1, used_host_e used_host=NOT_INIT_HOST_MEMORY) 
	: distributed_frame(distributed_frame_), state_count(state_count_), items_per_cell(items_per_cell_) {
		int dev_count = distributed_frame.get_dev_count();
		for(auto& frame : distributed_frame) {
			this->push_back(value_t<T>(frame, state_count_, items_per_cell, used_host));
		}
	}
	
	~distributed_value_t() {
		for(auto& value : *this) {
			value.destroy();
		}
	}
	
	void exchange() {
		exchange(state_count - 1);
	}
	
	void exchange(int state) {
		long byte_for_exchange = distributed_frame.get_items_for_exchange() * sizeof(T);
		auto beg = this->begin();
		for(auto it=beg; it!=this->end(); ++it) {
			if(it != this->begin()) {
				for(int j=0; j<items_per_cell; ++j) {					
					checkCudaErrors(
						cudaMemcpyPeer(
							it[-1].right_dst(state, j), std::distance(beg, it-1),
							(*it).left_src(state, j), std::distance(beg, it), 
							byte_for_exchange
						) 
					);
				}
			}
			
			if(it != std::prev(this->end())) {
				for(int j=0; j<items_per_cell; ++j) {					
					checkCudaErrors(
						cudaMemcpyPeer(
							it[1].left_dst(state, j), std::distance(beg, it+1), 
							(*it).right_src(state, j), std::distance(beg, it), 
							byte_for_exchange
						) 
					);
				}
			}
		}
	}
	
	void swap() {
		for(auto& frame : distributed_frame) {
			checkCudaErrors( cudaSetDevice(frame.dev_id) );
			checkCudaErrors( cudaDeviceSynchronize() );
		}
		
		for(auto& value : (*this)) {
			value.swap();
		}
	}
	
	void copyDtoH() {
		for(auto& value : *this) {
			value.copyDtoH();			
		}
	}
	
	void copyHtoD() {
		for(auto& value : *this) {
			value.copyHtoD();
		}
	}
	
	void copyDtoH(int state) {
		for(auto& value : *this) {
			value.copyDtoH(state);			
		}
	}
	
	void copyHtoD(int state) {
		for(auto& value : *this) {
			value.copyHtoD(state);
		}
	}
};



#endif//__CORE_DISTR_VAR_H__
#ifndef __CORE_DISTR_OUTPUT_H__
#define __CORE_DISTR_OUTPUT_H__

#include <vector>
#include <iostream>
#include <cassert>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <helper/helper_math.h>
#include <helper/helper_cuda.h>
#include "frame.h"

using std::cout;
using std::endl;

inline
int ceil_devide(int a, int b) {
	return (int) ceil(float(a) / float(b));
}

inline
int3 ceil_devide(int3 a, int3 b) {
	return make_int3(ceil_devide(a.x, b.x), ceil_devide(a.y, b.y), ceil_devide(a.z, b.z));
}


template <class T>
struct output_t {
	const frame_t& frame;
	T* gpu_pointer;
	int item_offset;
	int item_count;
	int3 dim;
	bool is_root;
	
// 	T* ptr() {
// 		return (gpu_pointer + is_root * item_offset);
// 	 }
	
	output_t(const frame_t& frame_, int item_offset_, int item_count_, int3 dim_, bool is_root_, T* gpu_pointer_ = nullptr)
	: frame(frame_), gpu_pointer(gpu_pointer_), item_offset(item_offset_), item_count(item_count_), dim(dim_), is_root(is_root_)
	{	}
	
	void destroy() {
		if(gpu_pointer) {
			checkCudaErrors( cudaSetDevice(frame.dev_id) );
			checkCudaErrors( cudaFree(gpu_pointer) );
			gpu_pointer = nullptr;
		}
	}
	
	operator T*() {
		return (gpu_pointer + is_root * item_offset);
	}
	
	T* move(int count) {
		return gpu_pointer + count;
	}
};

inline
std::ostream& operator<<(std::ostream& stream, int3 value) {
	stream << "[" << value.x << ", " << value.y << ", " << value.z << "]";
	return stream;
}


template <class T>
class distributed_output_t : public std::vector< output_t<T> >  {
	const distributed_frame_t& distributed_frame;
// 	std::vector<output_t<T> > outputs;
	T* host_pointer;
	int3 step;
	int3 _begin;
	int3 _end;	
	int3 dim;
	int item_count;
	
	output_t<T>* root;
	
public:
	distributed_output_t(const distributed_frame_t& distributed_frame_, int root_, 
						 int3 step_=make_int3(1, 1, 1), int3 begin_=make_int3(0, 0, 0), int3 end_=make_int3(-1, -1, -1))
	: distributed_frame(distributed_frame_), host_pointer(nullptr), step(step_), _begin(begin_), root(nullptr) {
		int3 main_dim = distributed_frame.get_dim();
		
		_end = make_int3(0, 0, 0);
		_end.x = (end_.x >= 0 and end_.x <= main_dim.x ? end_.x : main_dim.x);
		_end.y = (end_.y >= 0 and end_.y <= main_dim.y ? end_.y : main_dim.y);
		_end.z = (end_.z >= 0 and end_.z <= main_dim.z ? end_.z : main_dim.z);
		
		int3 cuted_main_dim = _end - _begin;
		dim = ceil_devide(cuted_main_dim, step);
// 		dim = make_int3 ( make_float3(cuted_main_dim) / make_float3(step) );
		item_count = dim.x * dim.y * dim.z;
		assert(dim.x > 0);
		assert(dim.y > 0);
		assert(dim.z > 0);
		
		if(root_ < 0) {
			host_pointer = new T[item_count];
// 			for(int i=0;i<item_count; ++i) {
// 				host_pointer[i] = {127, 127, 255, 255};
// 			}
			
			
		}

		int prev_x_items = 0;
		int layer = dim.y * dim.z;
		for(const auto& frame : distributed_frame) {
			
			int end_frame = frame.pos.x + frame.len.x;
			int real_end = _end.x < end_frame ? _end.x : end_frame;
			int3 cur_dim;
			if(_begin.x < frame.pos.x) {
				if(real_end <= frame.pos.x) {
					//make_stub
					assert(0);																	//FIXME
				}
				
				int pos_x = frame.pos.x - _begin.x;
				prev_x_items = ceil_devide(pos_x, step.x);
				int offset = prev_x_items * step.x - pos_x;
				int real_len = real_end - frame.pos.x - offset;
				
				assert(real_len > 0);															//FIXME
				cur_dim = make_int3(ceil_devide(real_len, step.x), dim.y, dim.z);
			}
			
			else if(_begin.x >= frame.pos.x and _begin.x < end_frame) {
				int real_len = real_end - _begin.x;
				cur_dim = make_int3(ceil_devide(real_len, step.x), dim.y, dim.z);
			}
			else {
				assert(0);																		//FIXME
			}	
			
			T* gpu_ptr = nullptr;
			bool is_root = (frame.dev_id == root_);
			checkCudaErrors( cudaSetDevice(frame.dev_id) );
			int byte_size = (is_root ? item_count : cur_dim.x * layer) * sizeof(T);
			checkCudaErrors( cudaMalloc((void **) &gpu_ptr, byte_size) );
			
			output_t<T> output(frame, prev_x_items * layer, cur_dim.x * layer, cur_dim, is_root, gpu_ptr);
			this->push_back(output);
			
		}
		if(root_ >= 0) root = &this->at(root_);
	}
	
	T* get_host_pointer() {
		return host_pointer;
	}
	
	
	~distributed_output_t() {
		if(host_pointer) {
			delete[] host_pointer;
		}
		
		for(auto& output : *this) {
			output.destroy();
		}		
	}
	
	void collect() {
		if(host_pointer) {
			for(auto& output : *this) {
// 				cout << "qq: " << output.frame.dev_id << " " << output.item_offset << " " << output.item_count << " " << sizeof(T) << endl;
				checkCudaErrors( cudaSetDevice(output.frame.dev_id) );
				checkCudaErrors( cudaMemcpy(host_pointer + output.item_offset, output, output.item_count * sizeof(T), cudaMemcpyDeviceToHost) );
			}
		}
		else {			
			for(auto& output : *this) {
				if(output.is_root) continue;
				checkCudaErrors(
					cudaMemcpyPeer(
						root->move(output.item_offset), root->frame.dev_id, output, output.frame.dev_id,
						output.item_count * sizeof(T)
					)
				);
			}
		}
	}
	
	int3 get_dim() const {
        return dim;
    }
	
};



#endif//__CORE_DISTR_OUTPUT_H__
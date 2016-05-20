#ifndef __CORE_FRAME_H__
#define __CORE_FRAME_H__

#include <vector>
#include <memory>
#include <cmath>

// #include "core.h"
#include <vector_types.h>
#include <helper/helper_math.h>
#include <iostream>

struct frame_t {
	int dev_id;
	long items_for_calculate;
	long items_for_exchange;
	long prev_items;
	long left;
	long right;
	
	bool has_left_neighbor;
	bool has_right_neighbor;
	int depth;
	
	int3 pos, len;
	
// 	frame_t(int dev_id_, long x_pos_, long items_for_calculate_, long items_for_exchange_, bool has_left_neighbor_, bool has_right_neighbor_)
// 	: dev_id(dev_id_), x_im_pos(x_pos_), items_for_calculate(items_for_calculate_), items_for_exchange(items_for_exchange_)
// 	, has_left_neighbor(has_left_neighbor_), has_right_neighbor(has_right_neighbor_)
// 	, left(has_left_neighbor_ * items_for_exchange_), right(has_right_neighbor_ * items_for_exchange_) {
// 		tid_offset = left;
// 		tid_end = left + items_for_calculate;
// 		throw 1;
// 	}
	
	frame_t(int dev_id_, int3 pos_, int3 len_, int depth_, bool has_left_neighbor_, bool has_right_neighbor_)
	: dev_id(dev_id_), pos(pos_), len(len_), depth(depth_), has_left_neighbor(has_left_neighbor_)
	, has_right_neighbor(has_right_neighbor_), items_for_exchange(len_.y * len_.z * depth_) {
		items_for_calculate = len.x * len.y * len.z;
		left = has_left_neighbor * items_for_exchange;
		right = has_right_neighbor * items_for_exchange;
		prev_items = pos.x * (len.y * len.z);
	}
	
	long get_item_count() const {		
		return items_for_calculate + left + right;
	}
	
	long offset_to_left_dst(int n) const {
		return get_item_count() * n;
	}
	
	long offset_to_left_src(int n) const {
		return offset_to_left_dst(n) + left;
	}
	
	long offset_to_right_dst(int n) const {
		return offset_to_left_src(n) + items_for_calculate;
	}
	
	long offset_to_right_src(int n) const {
		return offset_to_right_dst(n) - right;
	}
};

class distributed_frame_t : public std::vector<frame_t> {
	long items_for_exchange;
	int dev_count;
	int3 dim;
	
public:
	distributed_frame_t(int nx, int ny, int nz, int depth, int dev_count_)
	: distributed_frame_t({nx, ny, nz}, depth, dev_count_) { }
	
	distributed_frame_t(int3 dim_, int depth, int dev_count_) {
		dim = dim_;
		dev_count = dev_count_;
		int layer = dim.y * dim.z;
		items_for_exchange = depth * layer;
		
		for(int i=0; i<dev_count; ++i) {
// 			checkCudaErrors( cudaSetDevice(i) );				//TODO WTF???
			bool has_left_neighbor = ((i - 1) >= 0);
			bool has_right_neighbor = ((i + 1) < dev_count);
			int x_pos = dim.x / dev_count * i;
// 			int x_im_pos = x_pos - has_left_neighbor * depth;
			int x_len = dim.x / dev_count + (dim.x % dev_count) * (!has_right_neighbor);
// 			int items_for_calculate = (dim.x / dev_count + (dim.x % dev_count) * (!has_right_neighbor)) * layer;
			//frame_t frame(i, x_pos, items_for_calculate, items_for_exchange, has_left_neighbor, has_right_neighbor);
			
// 			frame_t(int dev_id_, int3 pos_, int3 lenght_, int depth_, bool has_left_neighbor_, bool has_right_neighbor_)
			frame_t frame(i, {x_pos, 0, 0}, {x_len, dim.y, dim.z}, depth, has_left_neighbor, has_right_neighbor);
			push_back(frame);
		}
	}
	
	int3 get_dim() const {
		return dim;
	}
	
	int get_dev_count() const {
		return dev_count;
	}
	
	long get_items_for_exchange() const {
		return items_for_exchange;
	}
	
// 	std::pair<int, int> get_block_and_thread_count(int id) const {
// 		float items = this->at(id).get_item_count();
// 		int block_count = (int) ceil(items / THREADS_PER_BLOCK);
// 		return std::pair<int, int>(block_count, THREADS_PER_BLOCK);
// 	}
	
// 	int get_thread_count(int id=-1) const {
// 		return THREADS_PER_BLOCK;
// 	}
// 	
// 	int get_block_count(int id) const {
// 		float items = (*this)[id].get_item_count();
// 		return (int) ceil(items / get_thread_count());
// 	}
};

#endif//__CORE_FRAME_H__
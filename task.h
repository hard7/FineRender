#ifndef __CORE_TASK_H__
#define __CORE_TASK_H__

#include <memory>

namespace core {

class Task {
    class ThreadImpl;
    
public:
    std::shared_ptr<ThreadImpl> threadImpl;
    
    virtual void iterate() =0;
    virtual void* getColorBuffer(int textureId =0) =0;
    virtual void fillColorBuffer(int textureId =0) =0;
    virtual void setDim(int& nx, int& ny, int& nz, int textureId =0) const =0;
    virtual ~Task() = default;
//    virtual inline bool using_host_color_buffer() { return true; };

    Task();
    void start();
    void setStepCount(int stepCount);
};

}//namespace core

#endif//__CORE_TASK_H__
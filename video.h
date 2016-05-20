#ifndef __CORE_VIDEO_H__
#define __CORE_VIDEO_H__

#include <iosfwd>
#include <memory>


namespace core {

class Video {
    class Impl;
    std::shared_ptr<Impl> impl;

public:
    Video(std::string const& path, int width, int height, double fps);

    unsigned char* getBuffer();
    void writeRGB(unsigned char const* frame = nullptr);
    void writeRGB0(unsigned char const* frame = nullptr);
};


}//core

#endif//__CORE_VIDEO_H__


#include "../video.h"
#include <string>
#include <vector>


#ifdef FFMPEG_FOUND

//
#ifndef UINT64_C
#define UINT64_C(c) (c ## ULL)
#endif
extern "C" {
#include <avcodec.h>
#include <avformat.h>
#include <swscale.h>
#include <log.h>
}

#include <stdexcept>
#include <iostream>
#include <map>

//using std::cout;
//using std::endl;

namespace {
    struct AV_Deleter {
        void operator()(AVFrame* frame) { av_free(frame); }
        void operator()(AVPacket* packet) { av_free_packet(packet); free(packet); }
        void operator()(AVStream* stream) { av_freep(stream); }
        void operator()(SwsContext* swCxt) { sws_freeContext(swCxt); }
        void operator()(AVFormatContext* ctx) {
            av_write_trailer(ctx);
            avformat_write_header(ctx, nullptr);
            avio_close(ctx->pb);
            av_free(ctx);
        }
    };
    template<typename T> using av_unique_ptr = std::unique_ptr<T, AV_Deleter>;
}
namespace core {

    struct Video::Impl {
        const int width, height;
        const double fps;
        uint32_t step;

        av_unique_ptr<AVPacket> packet;
        av_unique_ptr<AVFormatContext> formatContext;
        av_unique_ptr<AVStream> stream;
        av_unique_ptr<AVFrame> frame;
        std::map<AVPixelFormat, av_unique_ptr<SwsContext> > convertContexts;
        std::unique_ptr<uint8_t[]> picBuffer;
        std::unique_ptr<uint8_t[]> colorBoffer;


        static void initFFmpeg() {
#ifdef FFMPEG_QUIET
            static bool isQuilet = (av_log_set_level(AV_LOG_QUIET), true);
#endif
            static bool isInitialized = (av_register_all(), true);
        }

        Impl(std::string const& path, int width_, int height_, double fps_)
                : width(width_), height(height_), fps(fps_), step(0) {
            initFFmpeg();

            formatContext.reset(avformat_alloc_context());
            if(not formatContext) throw std::logic_error("Error allocating format context");

            formatContext->oformat = av_guess_format(nullptr, path.c_str(), nullptr);
            if (not formatContext->oformat) {
                std::cerr << "Could not deduce output format from file extension: using MPEG.";
                formatContext->oformat = av_guess_format("mpeg", nullptr, nullptr);
            }

            if(path.size() >= 1024) throw std::logic_error("Path to file can not to contain more than 1024 symbols");
            strcpy(formatContext->filename, path.c_str());

            if(avio_open(&formatContext->pb, path.c_str(), AVIO_FLAG_WRITE) < 0) {
                std::logic_error("Could not open file'" + path + "' for writing video");
            }

            stream.reset(avformat_new_stream(formatContext.get(), nullptr));
            if(not stream) std::logic_error("Could not allocate stream");
            stream->id = 0;
            stream->time_base = (AVRational){1, (int)fps};

            AVCodecContext *codecCtx = stream->codec;
            codecCtx->codec_id = formatContext->oformat->video_codec;       // AV_CODEC_ID_MPEG4;
            codecCtx->codec_type = AVMEDIA_TYPE_VIDEO;
            codecCtx->bit_rate = (int)(width * height * fps);
            codecCtx->width = width;
            codecCtx->height = height;
            codecCtx->time_base = (AVRational){1, (int)fps};
            codecCtx->gop_size = 1;
            codecCtx->pix_fmt = AV_PIX_FMT_YUV420P;

            codecCtx->qmin = 1;     // magic
            codecCtx->qmax = 31;

            if(formatContext->oformat->flags & AVFMT_GLOBALHEADER) codecCtx->flags |= CODEC_FLAG_GLOBAL_HEADER;
            av_dump_format(formatContext.get(), 0, path.c_str(), 1);

            AVCodec *codec = avcodec_find_encoder(codecCtx->codec_id);
            if(not codec) std::logic_error("Could not allocate stream");
            if(avcodec_open2(codecCtx, codec, nullptr) < 0) std::logic_error("Could not open codec");
            if(avformat_write_header(formatContext.get(), nullptr) < 0) std::logic_error("Couldn't write header into the stream");

            frame.reset(avcodec_alloc_frame());
            if(not frame) std::logic_error("Couldn't create frame");
            picBuffer.reset(new uint8_t[avpicture_get_size(codecCtx->pix_fmt, width, height)]);
            avpicture_fill(reinterpret_cast<AVPicture*>(frame.get()), picBuffer.get(), codecCtx->pix_fmt, width, height);
            frame->format = stream->codec->pix_fmt;
            frame->width = stream->codec->width;
            frame->height = stream->codec->height;

            packet.reset((AVPacket*)malloc(sizeof(AVPacket)));
            av_init_packet(packet.get());
        }

        void writeRGB(unsigned char const* dataPointer) { writeFrame(dataPointer, AV_PIX_FMT_RGB24); }
        void writeRGB0(unsigned char const* dataPointer) { writeFrame(dataPointer, AV_PIX_FMT_RGB0); }

        SwsContext* getContext(AVPixelFormat format) {
            if(convertContexts.find(format) == convertContexts.end()) {
                SwsContext* sc = sws_getContext(width, height, format, stream->codec->width, stream->codec->height,
                                                stream->codec->pix_fmt, SWS_BICUBLIN, nullptr, nullptr, nullptr);
                if(not sc) throw std::logic_error("Cannot initialize the conversion context");
                convertContexts[format].reset(sc);
            }
            return convertContexts[format].get();
        }

        void writeFrame(unsigned char const* dataPointer, AVPixelFormat format) {
            if(not dataPointer and not colorBoffer) throw std::logic_error("writeFrame without data");
            if(not dataPointer) dataPointer = colorBoffer.get();

            // http://stackoverflow.com/questions/15914012/encoding-a-screenshot-into-a-video-using-ffmpeg
            const uint8_t *slices[] = { dataPointer + width * 3 * (height-1) };
            int stride = width * 3 * (-1);
            sws_scale(getContext(format), slices, &stride, 0, height, frame->data, frame->linesize);
            AVPacket* packet = this->packet.get();
            
            frame->pts = ++step;          // * 2000; // * c->bit_rate / (c->time_base.den *c->time_base.den) ;
            int got_packet = 0;

            //https://ffmpeg.org/pipermail/libav-user/2013-June/004884.html
            av_free_packet(packet);
            av_init_packet(packet);

            if (avcodec_encode_video2(stream->codec, packet, frame.get(), &got_packet) < 0) {
                throw std::logic_error("Error occures duaring encoding video");
            }

            if(formatContext->oformat->flags & AVFMT_RAWPICTURE) {
                packet->flags |= AV_PKT_FLAG_KEY;
                packet->stream_index = stream->index;
            }
            else {
                packet->pts = av_rescale_q(step, stream->codec->time_base, stream->time_base);
                if(stream->codec->coded_frame->key_frame) packet->flags |= AV_PKT_FLAG_KEY;
                packet->stream_index = stream->index;
                packet->dts = packet->pts;
            }

            if(got_packet) av_interleaved_write_frame(formatContext.get(), packet);
        }

        unsigned char* getBuffer() {
            if(not colorBoffer) colorBoffer.reset(new uint8_t[width * height * 4]);
            return colorBoffer.get();
        }
    };

    Video::Video(std::string const& path, int width, int height, double fps) : impl(new Impl(path, width, height, fps)) {}
    void Video::writeRGB(unsigned char const* frame) { impl->writeRGB(frame); };
    void Video::writeRGB0(unsigned char const* frame) { impl->writeRGB0(frame); };
    unsigned char* Video::getBuffer() { return impl->getBuffer(); }

}// core

#else //FFMPEG_NOT_FOUND

#include <stdexcept>
static void ex(std::string const& what) { throw std::logic_error("[FFMPEG_NOT_FOUND] " + what); }

namespace core {
Video::Video(std::string const&, int, int, double) { ex("Video::Video(std::string const&, int, int, double)"); }
void Video::writeRGB(unsigned char const*) { ex("Video::writeRGB(unsigned char const*)"); }
void Video::writeRGB0(unsigned char const*) { ex("Video::writeRGB0(unsigned char const*)"); }
unsigned char* Video::getBuffer() { ex("Video::getBuffer()"); }
}

#endif


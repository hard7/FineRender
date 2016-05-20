#include "../render.h"
#include "render_impl.h"
#include "../task.h"
#include "task_impl.h"

#include <iostream>
#include <string>

namespace core {

Render::Render(Task& task, int width, int height, int textureId) {
    int nx = 1, ny = 1, nz = 1;
    task.setDim(nx, ny, nz, textureId);
    impl.reset(new Impl(nx, ny, nz, task.threadImpl->semaphore, textureId));
    impl->resize(width, height);
// 	QObject::connect(task.thread_impl, SIGNAL( signal_update() ), impl, SLOT( slot_update() ));
    QObject::connect(impl.get(), SIGNAL( signal_closed_render(Render::Impl*) ), task.threadImpl.get(), SLOT( slot_disconnect_client(Render::Impl*) ));
    QObject::connect(impl.get(), SIGNAL( signal_change_frequency(Render::Impl*, int) ), task.threadImpl.get(), SLOT( slot_change_frequency(Render::Impl*, int) ));
    task.threadImpl->client_count++;
    setFrequency(100);
    impl->show();
}

Render::Render(Task& task, int textureId) : Render(task, 640, 480, textureId) { }
void Render::setBaseRotate(int x, int y, int z) { impl->set_base_rotate(x, y, z); }
void Render::setFrequency(int freq) {impl->set_frequency(freq); }
void Render::setDepthLayers(int s) { impl->set_depth_layers(s); }
void Render::enableVideo(std::string const& path, double fps) { impl->enableVideo(path, fps); }
	
}//namespace core

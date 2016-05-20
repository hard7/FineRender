#ifndef __CORE_TASK_IMPL_H__
#define __CORE_TASK_IMPL_H__

#include "render_impl.h"
#include "../task.h"
#include <QObject>
#include <QThread>
#include <QApplication>
#include <QSemaphore>

#include <iostream>
#include <map>
#include <vector>

namespace core {
	
class Task::ThreadImpl : public QObject {
	Q_OBJECT
	
	Task* base;
	QThread thread;
	QSemaphore semaphore;
	int client_count;
	int step_count;
	bool is_active;
	
	std::map<Render::Impl*, int> render_to_freq;
	std::map<int, std::vector<Render::Impl*> > freq_to_renders;
	
	friend class Render;
	friend class Task;
	
public:
	ThreadImpl(Task* base_) : base(base_) {
		step_count = INT_MAX;
		client_count = 0;
		is_active = true;
		QObject::connect(this, SIGNAL( signal_finished() ), &thread, SLOT( quit() ));
		QObject::connect(&thread, SIGNAL( finished() ), QApplication::instance(), SLOT( quit() ));

	}
	
	void make_output(int step) {
		int counter = 0;
		for(auto&  item : freq_to_renders) {
			if(step % item.first == 0) {
				for(auto render : item.second) {
					int id = render->get_texture_id();
					base->fillColorBuffer(id);
					QMetaObject::invokeMethod(render, "slot_update", Q_ARG(void*, base->getColorBuffer(id)));
					counter++;
				}
			}
		}
		semaphore.acquire(counter);
	}
	
	
public slots:
	void slot_run() {
		for(int i=0; is_active and i<step_count; ++i) {
			base->iterate();
			make_output(i);
			QApplication::processEvents();
		}
		
		emit signal_finished();
	}
	
	
	
	void slot_disconnect_client(Render::Impl* render) {
		int freq = render_to_freq[render];
		std::vector<Render::Impl*>& vec = freq_to_renders[freq];
		vec.erase(std::find(std::begin(vec), std::end(vec), render));
		render_to_freq.erase(render);
		
		if(render_to_freq.empty()) is_active = false;
	}
	
	void slot_change_frequency(Render::Impl* render, int frequency) {
		if(render_to_freq.find(render) == std::end(render_to_freq)) {
			render_to_freq[render] = frequency;
			freq_to_renders[frequency].push_back(render);
		}
		else {
			int old_freq = render_to_freq[render];
			if(old_freq != frequency) {
				render_to_freq[render] = frequency;
				std::vector<Render::Impl*>& old = freq_to_renders[old_freq];
				old.erase(std::find(std::begin(old), std::end(old), render));
				freq_to_renders[frequency].push_back(render);
			}
		}
	} 			
	
	
signals:
	void signal_update();
	void signal_finished();
};


}//namespace core

#endif//__CORE_TASK_IMPL_H__
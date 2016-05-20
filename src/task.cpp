#include "../task.h"
#include "task_impl.h"

#include <QTimer>

namespace core {
	
Task::Task() : threadImpl(new ThreadImpl(this)) { }

void Task::start() {
	threadImpl->moveToThread( &threadImpl->thread );
	threadImpl->thread.start();
	QTimer::singleShot(0, threadImpl.get(), SLOT( slot_run() ));
}

void Task::setStepCount(int n) { threadImpl->step_count = n; }
	
}//namespace core
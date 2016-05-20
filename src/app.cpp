#include "../render.h"

#include <QApplication>
#include <QTextCodec>

namespace core {	
	
struct App::Impl {
	QApplication app;
	Impl(int &argc, char **argv) : app(argc, argv) {
		app.setQuitOnLastWindowClosed(false);
		QTextCodec::setCodecForTr(QTextCodec::codecForName("UTF8"));
	}
};

App::App(int &argc, char **argv) : impl(new Impl(argc, argv)) {  }

int App::exec() { return QApplication::exec(); }

}//namespace core

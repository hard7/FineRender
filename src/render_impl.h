#ifndef __CORE_RENDER_IMPL_H__
#define __CORE_RENDER_IMPL_H__


#include "../render.h"
#include "../task.h"
#include "../video.h"

#include <GL/glut.h>
#include <QGLWidget>
#include <QGLFunctions>
#include <QtTest/QTest>
#include <QSemaphore>
#include <QMatrix4x4>
#include <QSemaphore>

#include <cmath>

#include <iostream>

namespace core {

class Render::Impl : public QGLWidget, protected QGLFunctions  {
	Q_OBJECT
	
	const int tx;
	
	GLuint texture;
	int dim_x, dim_y, dim_z;
	QMatrix4x4 M_model, M_rot, M_base_rot, M_proj;
	QPoint currentPos;
	float item_per_pixel;
	float global_scale;
	int texture_layer;

    std::shared_ptr<Video> video;
	
	QSemaphore& semaphore;
	
	friend class Render;
	
public:
	Impl(int nx, int ny, int nz, QSemaphore& semaphore_, int tx_)
	: texture(0), tx(tx_), semaphore(semaphore_) {
		dim_x = nx;
		dim_y = ny;
		dim_z = nz;
		global_scale = 1;
		texture_layer = TEXTURE_LAYER_Z;
	}
	
	void set_depth_layers(int s) { texture_layer = s; }
	
	void closeEvent(QCloseEvent* event) {
		emit signal_closed_render(this);
	}
	
	inline void initializeGL() override {
		glClearColor(0, 0, 0, 0);
		glEnable(GL_DEPTH_TEST);
        glEnable(GL_TEXTURE_3D);
        glEnable(GL_ALPHA_TEST);
        glEnable(GL_BLEND);
		
//         glDepthMask(GL_FALSE);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_3D, texture);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_3D, 0);
		
		reset_state();
	}
	
	int get_texture_id() {
		return tx;
	}
	
	void reset_state() {
		M_model.setToIdentity();
		M_rot.setToIdentity();
		item_per_pixel *= global_scale;
		global_scale = 1;
		M_rot = M_base_rot;
	}
	
	inline void resizeGL(int width, int height) override {
		if (height == 0) height = 1;
		GLfloat aspect = (GLfloat)width / (GLfloat)height;
		glViewport(0, 0, width, height);
		
		GLfloat window_aspect = (GLfloat)width / (GLfloat)height;
		GLfloat task_aspect = (GLfloat)dim_x / (GLfloat)dim_z;
		
		float w, h;
		float d = sqrt(dim_x * dim_x + dim_y * dim_y + dim_z * dim_z);
		if(window_aspect > 1) {
			w = (h = d) * window_aspect;
		}
		else {
			h = (w = d) / window_aspect;
		}
		
		
		M_proj.setToIdentity();
		M_proj.ortho(-w/2, w/2, -h/2, h/2, -d * 10, d/2);
		
		item_per_pixel = h / height / global_scale;
// 		glOrtho(-w/2, w/2, -h/2, h/2, -d * 10, d/2);
	}
	
	
	inline void paintGL() override {		
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glMatrixMode(GL_PROJECTION);
		glLoadMatrixd(M_proj.data());
		
		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixd(M_model.data());
		glMultMatrixd(M_rot.data());
		
		glBindTexture(GL_TEXTURE_3D, texture);
		
		glBegin(GL_QUADS); 
		
		float x = dim_x,y = dim_y, z = dim_z;
		if(texture_layer & TEXTURE_LAYER_Y) {
			float depth = fmin(y, 30);
			float step = 1 / depth;
		
			for(float i=0; i<1+step/2; i += step) {
				float cur_y = (i - 0.5) * y;
				glTexCoord3f(1, i, 0); glVertex3f(-x/2, cur_y, -z/2);
				glTexCoord3f(1, i, 1); glVertex3f( x/2, cur_y, -z/2);
				glTexCoord3f(0, i, 1); glVertex3f( x/2, cur_y,  z/2);
				glTexCoord3f(0, i, 0); glVertex3f(-x/2, cur_y,  z/2);
			}
		}
		
		if(texture_layer & TEXTURE_LAYER_Z) {
			float depth = fmin(z, 30);
			float step = 1 / depth;
		
			for(float i=0; i<1+step/2; i += step) {
				float cur_z = (i - 0.5) * z;
				glTexCoord3f(1-i, 0, 0); glVertex3f(-x/2, -y/2, cur_z);
				glTexCoord3f(1-i, 0, 1); glVertex3f( x/2, -y/2, cur_z);
				glTexCoord3f(1-i, 1, 1); glVertex3f( x/2,  y/2, cur_z);
				glTexCoord3f(1-i, 1, 0); glVertex3f(-x/2,  y/2, cur_z);
			}
		}
		
		if(texture_layer & TEXTURE_LAYER_X) {
			float depth = fmin(x, 30);
			float step = 1 / depth;
		
			for(float i=0; i<=1+step/2; i += step) {
				float cur_x = (0.5 - i ) * x;
				glTexCoord3f(1, 0, 1-i); glVertex3f(cur_x, -y/2, -z/2);
				glTexCoord3f(0, 0, 1-i); glVertex3f(cur_x, -y/2,  z/2);
				glTexCoord3f(0, 1, 1-i); glVertex3f(cur_x,  y/2,  z/2);
				glTexCoord3f(1, 1, 1-i); glVertex3f(cur_x,  y/2, -z/2);
			}
		}
		
		glEnd();
		glBindTexture(GL_TEXTURE_3D, 0);
		
		glLoadIdentity();
		glViewport(0, 0, width(), 50);	
		
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		
		
////        cube
//		glBegin(GL_QUADS);
//		glColor4f(1, 0, 0, 0.4);
//
//		glVertex3f(-1, -1, 0);
//		glVertex3f( 1, -1, 0);
//		glVertex3f( 1,  1, 0);
//		glVertex3f(-1,  1, 0);
//
//		glColor4f(1, 1, 1, 1);
//		glEnd();

////        text
//		glDisable(GL_TEXTURE_3D);
//		QFont font("Arial");
//		qglColor(Qt::white);
//		renderText(0, height() - 20, tr("Awesome! Yeah! Fairly straight forward"));

		glEnable(GL_TEXTURE_3D);
		
		glViewport(0, 0, width(), height());
		
		swapBuffers();
	}
	
	void mousePressEvent(QMouseEvent *event) override {
		currentPos = event->pos();
		if(event->buttons() & Qt::MiddleButton) {			
			reset_state();
			updateGL();
		}
	}
	
	void mouseMoveEvent(QMouseEvent *event) override {
		if(event->buttons() & Qt::LeftButton) {    
			float rot_x = (event->x() - currentPos.x()) * 180.0f / (global_scale * width());
			float rot_y = (event->y() - currentPos.y()) * 180.0f / (global_scale * height());
			
			M_rot.rotate(rot_y, M_rot.row(0).toVector3D());
			M_rot.rotate(rot_x, M_rot.row(1).toVector3D());
		}
		
		if(event->buttons() & Qt::RightButton) {
			float tr_x = (event->x() - currentPos.x()) * item_per_pixel;
			float tr_y = (event->y() - currentPos.y()) * item_per_pixel;
			M_model.translate(tr_x, -tr_y, 0);			
		}
		
		currentPos = event->pos();
		updateGL();
		

	}
	
	void wheelEvent(QWheelEvent *event) override {
		makeCurrent();
		
		float delta = event->delta() / 120;
		float scale = pow(2 , delta / 4);
		item_per_pixel /= scale;
		global_scale *= scale;
		
		glLoadMatrixd(M_model.data());
		GLdouble winX = event->x();
		GLdouble winY = height() - event->y();
		
		GLint Mview[4];
		GLdouble Mmodel[16], Mproj[16], x, y, z;
		glGetDoublev(GL_MODELVIEW_MATRIX, Mmodel);
		glGetDoublev(GL_PROJECTION_MATRIX, Mproj);
		glGetIntegerv(GL_VIEWPORT, Mview);
		gluUnProject(winX, winY, 0, Mmodel, Mproj, Mview, &x, &y, &z);
		
		M_model.translate(x, y, 0);
		M_model.scale(scale, scale, scale);
		M_model.translate(-x, -y, 0);
		
		updateGL();
	}
	
	inline void set_base_rotate(int x, int y, int z) {
		M_base_rot.setToIdentity();
		M_base_rot.rotate(x, M_rot.row(0).toVector3D());
		M_base_rot.rotate(y, M_rot.row(1).toVector3D());
		M_base_rot.rotate(z, M_rot.row(2).toVector3D());
		reset_state();
		updateGL();
	}
	
	void set_frequency(int freq) {
		emit signal_change_frequency(this, freq);
	}

    void enableVideo(std::string const& path, double fps) {
        video.reset(new Video(path, width(), height(), fps));
		setFixedSize(width(), height());
    }
	
public slots:
	void slot_update(void* ptr) {		
		makeCurrent();
		glBindTexture(GL_TEXTURE_3D, texture);
		glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, dim_z, dim_y, dim_x, 0, 
					GL_RGBA, GL_UNSIGNED_BYTE, ptr);
		glBindTexture(GL_TEXTURE_3D, 0);
		semaphore.release();
		updateGL();

		if(video) {
			glReadPixels(0, 0, width(), height(), GL_RGB, GL_UNSIGNED_BYTE, video->getBuffer());
			video->writeRGB();
		}
	}
	
signals:
	void signal_closed_render(Render::Impl*);
	void signal_change_frequency(Render::Impl*, int);
};
	
}


#endif//__CORE_RENDER_IMPL_H__

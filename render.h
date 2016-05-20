#ifndef __CORE_RENDER_H__
#define __CORE_RENDER_H__

#include <iosfwd>
#include <memory>

namespace core {
    class Task;
    
    class Render {
    public:
        class Impl;
        std::shared_ptr<Impl> impl;

		static const int TEXTURE_LAYER_X = 1;
		static const int TEXTURE_LAYER_Y = 2; 
		static const int TEXTURE_LAYER_Z = 4;
		
        Render(Task&, int textureId =0);
        Render(Task&, int width, int height, int textureId =0);

		void setBaseRotate(int x, int y, int z);
		void setFrequency(int);
		void setDepthLayers(int);

		void enableVideo(std::string const& path, double fps);
    };
		
	
	class App {
		class Impl;
        std::shared_ptr<Impl> impl;
	
	public:
		App(int &argc, char **argv);
		int exec();
	};
}

#endif//__CORE_RENDER_H__

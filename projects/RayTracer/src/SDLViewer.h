#pragma once

#include "Film.h"
#include "Renderer.h"

class SDLViewer
{
public:
    SDLViewer(const std::shared_ptr<Film>& film, Renderer* renderer, const char* title = "RayTracer");
    ~SDLViewer();

    bool Show(const std::shared_ptr<Film>& film) const;
    void Loop();
private:
    int _width;
    int _height;

    std::shared_ptr<Film> _film;
    Renderer* _renderer;

    struct SDL_Window* _window = nullptr;
    struct SDL_Renderer* _sdlRenderer = nullptr;
    struct SDL_Texture* _texture = nullptr;
    bool _initialized = false;
    bool running = true;
};

#include "SDLViewer.h"
#include <SDL3/SDL.h>
#include <chrono>
#include <iomanip>

using hi_clock = std::chrono::high_resolution_clock;

SDLViewer::SDLViewer(const std::shared_ptr<Film>& film, Renderer* renderer, const char* title)
    : _width(film->GetTamX()), _height(film->GetTamY()), _film(film), _renderer(renderer)
{
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        return;
    }

    if (!SDL_CreateWindowAndRenderer(title, _width, _height, SDL_WINDOW_RESIZABLE, &_window, &_sdlRenderer))
    {
        SDL_Log("SDL_CreateWindowAndRenderer failed: %s", SDL_GetError());
        SDL_Quit();
        return;
    }

    _texture = SDL_CreateTexture(
        _sdlRenderer,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        _width,
        _height);

    if (!_texture)
    {
        SDL_Log("SDL_CreateTexture failed: %s", SDL_GetError());
        SDL_DestroyRenderer(_sdlRenderer);
        SDL_DestroyWindow(_window);
        SDL_Quit();
        _sdlRenderer = nullptr;
        _window = nullptr;
        return;
    }

    _initialized = true;
}

SDLViewer::~SDLViewer()
{
    if (_texture)
        SDL_DestroyTexture(_texture);

    if (_sdlRenderer)
        SDL_DestroyRenderer(_sdlRenderer);

    if (_window)
        SDL_DestroyWindow(_window);

    if (_initialized)
        SDL_Quit();
}

bool SDLViewer::Show(const std::shared_ptr<Film>& film) const
{
    if (!_initialized)
        return false;

    if (!SDL_UpdateTexture(_texture, nullptr, film->Data(), film->Pitch()))
        SDL_Log("SDL_UpdateTexture failed: %s", SDL_GetError());

    SDL_RenderClear(_sdlRenderer);
    SDL_RenderTexture(_sdlRenderer, _texture, nullptr, nullptr);
    SDL_RenderPresent(_sdlRenderer);

    return true;
}

void SDLViewer::Loop()
{
    if (!_initialized)
        return;

    while (running)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_EVENT_QUIT)
            {
                running = false;
            }
            else if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE)
            {
                running = false;
            }
        }

        auto start = hi_clock::now();
        _renderer->Render();
        auto end = hi_clock::now();
        std::chrono::duration<float> renderTime = end - start;

        const float renderSeconds = renderTime.count();
        const float renderMs = renderSeconds * 1000.0f;
        const float fps = (renderSeconds > 0.0f) ? (1.0f / renderSeconds) : 0.0f;

        if (!Show(_film))
            running = false;

        SDL_Log("Render: %.4f ms | FPS: %.2f", renderMs, fps);
    }
}
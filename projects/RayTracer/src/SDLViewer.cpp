#include "SDLViewer.h"
#include <SDL3/SDL.h>
#include <chrono>
#include <iomanip>
#include "imgui.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_sdlrenderer3.h"
#include <filesystem>

using hi_clock = std::chrono::high_resolution_clock;

namespace
{
const char* GetBuildPlatform()
{
#ifdef _WIN64
    return "Win64";
#else
    return "Other";
#endif
}

const char* GetBuildConfiguration()
{
#ifdef _DEBUG
    return "Debug";
#else
    return "Release";
#endif
}
}

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

    const auto metricsPath = std::filesystem::current_path() / "metrics.csv";
    const bool writeHeader = !std::filesystem::exists(metricsPath);
    _metricsFile.open(metricsPath, std::ios::out | std::ios::app);
    if (_metricsFile.is_open() && writeHeader)
        _metricsFile << "backend,platform,configuration,width,height,render_ms,frame_ms,render_fps,frame_fps\n";

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    if (!ImGui_ImplSDL3_InitForSDLRenderer(_window, _sdlRenderer))
    {
        SDL_Log("ImGui_ImplSDL3_InitForSDLRenderer failed");
        return;
    }
    if (!ImGui_ImplSDLRenderer3_Init(_sdlRenderer))
    {
        SDL_Log("ImGui_ImplSDLRenderer3_Init failed");
        ImGui_ImplSDL3_Shutdown();
        return;
    }

    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = 2.0f;
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(2.0f);

    _imguiInitialized = true;
}

SDLViewer::~SDLViewer()
{
    if (_metricsFile.is_open())
        _metricsFile.close();

    if (_imguiInitialized)
    {
        ImGui_ImplSDLRenderer3_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
    }

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

    if (_imguiInitialized)
    {
        ImGui_ImplSDLRenderer3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        ImGui::ShowMetricsWindow();

        ImGui::Render();
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), _sdlRenderer);
    }

    SDL_RenderPresent(_sdlRenderer);

    return true;
}

void SDLViewer::Loop()
{
    if (!_initialized)
        return;

    while (running)
    {
        auto frameStart = hi_clock::now();

        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (_imguiInitialized)
                ImGui_ImplSDL3_ProcessEvent(&event);

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
        const float fpsRenderOnly = (renderSeconds > 0.0f) ? (1.0f / renderSeconds) : 0.0f;

        if (!Show(_film))
            running = false;

        auto frameEnd = hi_clock::now();
        std::chrono::duration<float> frameTime = frameEnd - frameStart;
        const float frameSeconds = frameTime.count();
        const float frameMs = frameSeconds * 1000.0f;
        const float fpsFrameTotal = (frameSeconds > 0.0f) ? (1.0f / frameSeconds) : 0.0f;

        if (_metricsFile.is_open())
        {
            _metricsFile
                << _renderer->GetBackendUsed() << ','
                << GetBuildPlatform() << ','
                << GetBuildConfiguration() << ','
                << _width << ','
                << _height << ','
                << renderMs << ','
                << frameMs << ','
                << fpsRenderOnly << ','
                << fpsFrameTotal << '\n';
            _metricsFile.flush();
        }

        SDL_Log("backend = %s | render_ms = %.4f | frame_ms = %.4f | render_fps = %.2f | frame_fps = %.2f",_renderer->GetBackendUsed(), renderMs, frameMs, fpsRenderOnly, fpsFrameTotal);
    }
}

#pragma once
#include "Camera.hpp"
#include "Film.h"
#include "Shape.h"
#include "World.h"

#include <memory>

class CUDABackend;

enum class BackendUsed
{
	CPU,
	CUDA
};

class Renderer
{
public:
	Renderer(std::shared_ptr<Film> film, std::shared_ptr<Camera> camera, std::shared_ptr<World> world, bool cudaEnabled = true);
	~Renderer();

	void Render(); // genera la escena
	const char* GetBackendUsed() const;
	Color RayColor(const Ray& r, int k); // devuelve el color del rayo lanzado sobre la geometria
private:
	void RenderCPU();
	std::shared_ptr<Film> _film;
	std::shared_ptr<Camera> _camera;
	std::shared_ptr<World> _world;
	std::unique_ptr<CUDABackend> _cudaBackend;
	BackendUsed _backendUsed = BackendUsed::CPU;
};

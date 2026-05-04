#include "Renderer.h"

#include "CheckerTexture.h"
#include "ConstantTexture.h"
#include "DirectionalLight.h"
#include "glm/geometric.hpp"
#include "PointLight.h"
#include "Quad.h"
#include "Scene.h"
#include "Sphere.h"
#include "Texture.h"
#include "cuda/CUDABackend.h"

#include <unordered_map>
#include <vector>

int FlattenTexture(
	const std::shared_ptr<Texture>& texture,
	std::vector<CudaTexture>& textures,
	std::unordered_map<const Texture*, int>& textureIndices)
{
	if (texture == nullptr)
		return -1;

	const auto found = textureIndices.find(texture.get());
	if (found != textureIndices.end())
		return found->second;

	const int index = static_cast<int>(textures.size());
	textureIndices[texture.get()] = index;
	textures.emplace_back();

	if (auto constant = std::dynamic_pointer_cast<ConstantTexture>(texture))
	{
		textures[index].type = CUDA_TEXTURE_CONSTANT;
		textures[index].color = constant->GetStoredColor();
		return index;
	}

	if (auto checker = std::dynamic_pointer_cast<CheckerTexture>(texture))
	{
		textures[index].type = CUDA_TEXTURE_CHECKER;
		textures[index].texture1 = FlattenTexture(checker->GetTexture1(), textures, textureIndices);
		textures[index].texture2 = FlattenTexture(checker->GetTexture2(), textures, textureIndices);
		textures[index].rows = checker->GetRows();
		textures[index].columns = checker->GetColumns();
		return index;
	}

	return -1;
}

int FlattenMaterial(
	const std::shared_ptr<Material>& material,
	std::vector<CudaMaterial>& materials,
	std::vector<CudaTexture>& textures,
	std::unordered_map<const Material*, int>& materialIndices,
	std::unordered_map<const Texture*, int>& textureIndices)
{
	const auto found = materialIndices.find(material.get());
	if (found != materialIndices.end())
		return found->second;

	CudaMaterial flatMaterial;
	flatMaterial.color = material->GetBaseColor();
	flatMaterial.glossFactor = material->GetGlossFactor();
	flatMaterial.textureIndex = FlattenTexture(material->GetTexture(), textures, textureIndices);

	const int index = static_cast<int>(materials.size());
	materialIndices[material.get()] = index;
	materials.push_back(flatMaterial);

	return index;
}

void FlattenShape(
	const std::shared_ptr<Shape>& shape,
	std::vector<CudaShape>& shapes,
	std::vector<CudaMaterial>& materials,
	std::vector<CudaTexture>& textures,
	std::unordered_map<const Material*, int>& materialIndices,
	std::unordered_map<const Texture*, int>& textureIndices)
{
	if (auto scene = std::dynamic_pointer_cast<Scene>(shape))
	{
		for (const auto& child : scene->GetShapes())
			FlattenShape(child, shapes, materials, textures, materialIndices, textureIndices);
		return;
	}

	CudaShape flatShape;

	if (auto sphere = std::dynamic_pointer_cast<Sphere>(shape))
	{
		flatShape.type = CUDA_SHAPE_SPHERE;
		flatShape.center = sphere->GetCenter();
		flatShape.radius = sphere->GetRadius();
		flatShape.materialIndex = FlattenMaterial(sphere->GetMaterial(), materials, textures, materialIndices,
		                                          textureIndices);
		shapes.push_back(flatShape);
		return;
	}

	if (auto quad = std::dynamic_pointer_cast<Quad>(shape))
	{
		flatShape.type = CUDA_SHAPE_QUAD;
		flatShape.q = quad->GetQ();
		flatShape.u = quad->GetU();
		flatShape.v = quad->GetV();
		flatShape.materialIndex = FlattenMaterial(quad->GetMaterial(), materials, textures, materialIndices,
		                                          textureIndices);
		shapes.push_back(flatShape);
	}
}

std::vector<CudaLight> FlattenLights(const std::vector<std::shared_ptr<Light>>& lights)
{
	std::vector<CudaLight> flatLights;
	flatLights.reserve(lights.size());

	for (const auto& light : lights)
	{
		CudaLight flatLight;
		flatLight.shadow = light->GetShadow() ? 1 : 0;
		flatLight.glossPower = light->GetGlossPower();

		if (auto point = std::dynamic_pointer_cast<PointLight>(light))
		{
			flatLight.type = CUDA_LIGHT_POINT;
			flatLight.position = point->GetPosition();
			flatLight.color = point->GetColor();
			flatLights.push_back(flatLight);
			continue;
		}

		if (auto directional = std::dynamic_pointer_cast<DirectionalLight>(light))
		{
			flatLight.type = CUDA_LIGHT_DIRECTIONAL;
			flatLight.direction = directional->GetDirection();
			flatLight.color = directional->GetColor();
			flatLights.push_back(flatLight);
		}
	}

	return flatLights;
}

Renderer::Renderer(std::shared_ptr<Film> film, std::shared_ptr<Camera> camera, std::shared_ptr<World> world, bool cudaEnabled)
	: _film(film), _camera(camera), _world(world)
{
	if (!cudaEnabled) return;

	std::vector<CudaShape> shapes;
	std::vector<CudaMaterial> materials;
	std::vector<CudaTexture> textures;
	std::unordered_map<const Material*, int> materialIndices;
	std::unordered_map<const Texture*, int> textureIndices;

	FlattenShape(_world->GetScene(), shapes, materials, textures, materialIndices, textureIndices);

	const CudaCamera flatCamera{
		_camera->GetPosition(),
		_camera->GetDeltaX(),
		_camera->GetDeltaY(),
		_camera->GetPositionTopLeft()
	};

	_cudaBackend = std::make_unique<CUDABackend>(
		_film->GetTamX(),
		_film->GetTamY(),
		flatCamera,
		shapes,
		materials,
		textures,
		FlattenLights(_world->GetLights()));

	_backendUsed = BackendUsed::CUDA;
}

Renderer::~Renderer() {}

void Renderer::Render()
{
	if (_cudaBackend != nullptr && _cudaBackend->Render(_film->MutableData()))
		return;
	RenderCPU();
}

const char* Renderer::GetBackendUsed() const
{
	return (_backendUsed == BackendUsed::CUDA) ? "CUDA" : "CPU";
}

void Renderer::RenderCPU()
{
	for (int y = 0; y < _film->GetTamY(); ++y)
	{
		for (int x = 0; x < _film->GetTamX(); ++x)
		{
			//std::cout << "Escribiendo pixel numero " << x << " " << y << " de ancho " << _film->GetTamX() << " e y " << _film->GetTamY() << ".\n";
			const Ray ray_primary = _camera->GetRay(x, y);
			const Color c = RayColor(ray_primary, 10);
			_film->AddPixel(x, y, c);
		}
	}
}

Color Renderer::RayColor(const Ray& r, int k)
{
	if (k <= 0) return BLACK;

	InfoIntersection ii;
	if (_world->GetScene()->Intersect(r, 0.001f, 100, ii))
	{
		Color color = BLACK;
		color += (ii.m->GetColor(ii) * Color(0.1, 0.1, 0.1)); // Luz de ambiente cableada

		for (auto& l : _world->GetLights())
		{
			if (l->GetShadow())
			{
				glm::vec3 origin = ii.p;
				glm::vec3 dir = l->ShadowDir(ii.p);
				float maxDist = glm::length(dir);
				dir = glm::normalize(dir);

				Ray shadowRay(origin, dir);

				if (_world->GetScene()->Intersect(shadowRay, 0.001f, maxDist))
				{
					// saltar a la siguiente geometria
					continue;
				}
			}
			color += l->Shade(r, ii);
		}

		if (ii.m->GetGlossFactor() > 0.0f && k > 0)
		{
			glm::vec3 hitPos = ii.p + ii.normal * 0.001f;
			const glm::vec3 normal = glm::normalize(ii.normal);
			const glm::vec3 dir = glm::reflect(glm::normalize(r.Direction()), normal);

			Ray shadowRay(hitPos, dir);

			color += ii.m->GetGlossFactor() * RayColor(shadowRay, k - 1);
		}

		return color;
	}

	// Skybox
	return BLACK;
}

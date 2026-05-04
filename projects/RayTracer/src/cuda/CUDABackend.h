#pragma once

#include "glm/vec3.hpp"

#include <vector>

using vec3 = glm::vec3;

struct CudaCamera
{
	vec3 position;
	vec3 deltaX;
	vec3 deltaY;
	vec3 positionTopLeft;
};

enum CudaShapeType
{
	CUDA_SHAPE_SPHERE = 0,
	CUDA_SHAPE_QUAD = 1
};

struct CudaShape
{
	int type = CUDA_SHAPE_SPHERE;
	int materialIndex = -1;

	vec3 center;
	float radius = 0.0f;

	vec3 q;
	vec3 u;
	vec3 v;
};

enum CudaTextureType
{
	CUDA_TEXTURE_CONSTANT = 0,
	CUDA_TEXTURE_CHECKER = 1
};

struct CudaTexture
{
	int type = CUDA_TEXTURE_CONSTANT;
	vec3 color;
	int texture1 = -1;
	int texture2 = -1;
	int rows = 1;
	int columns = 1;
};

struct CudaMaterial
{
	vec3 color;
	float glossFactor = 0.0f;
	int textureIndex = -1;
};

enum CudaLightType
{
	CUDA_LIGHT_POINT = 0,
	CUDA_LIGHT_DIRECTIONAL = 1
};

struct CudaLight
{
	int type = CUDA_LIGHT_POINT;
	int shadow = 0;
	int glossPower = 400;
	vec3 position;
	vec3 direction;
	vec3 color;
};

class CUDABackend
{
public:
	CUDABackend(
		int width,
		int height,
		const CudaCamera& camera,
		const std::vector<CudaShape>& shapes,
		const std::vector<CudaMaterial>& materials,
		const std::vector<CudaTexture>& textures,
		const std::vector<CudaLight>& lights);
	~CUDABackend();

	bool IsValid() const { return _valid; }
	bool Render(unsigned char* hostPixels);

private:
	int _width = 0;
	int _height = 0;
	bool _valid = false;
	int _shapeCount = 0;
	int _materialCount = 0;
	int _textureCount = 0;
	int _lightCount = 0;

	CudaCamera* _deviceCamera = nullptr;
	CudaShape* _deviceShapes = nullptr;
	CudaMaterial* _deviceMaterials = nullptr;
	CudaTexture* _deviceTextures = nullptr;
	CudaLight* _deviceLights = nullptr;
	unsigned char* _devicePixels = nullptr;
};

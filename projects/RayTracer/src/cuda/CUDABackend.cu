#include "CUDABackend.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "glm/geometric.hpp"

#include <cmath>
#include <corecrt_math_defines.h>



__device__ vec3 rayAt(const vec3& origin, const vec3& direction, float t)
{
	return origin + direction * t;
}

struct HitInfo
{
	vec3 p;
	vec3 normal;
	float t = 0.0f;
	float u = 0.0f;
	float v = 0.0f;
	int materialIndex = -1;
};

__device__ void sphereUv(const vec3& p, float& u, float& v)
{
	const float theta = acosf(-p.y);
	const float phi = atan2f(-p.z, p.x) + M_PI;

	u = phi / (2.0f * M_PI);
	v = theta / M_PI;
}

__device__ bool intersectSphere(
	const CudaShape& shape,
	const vec3& origin,
	const vec3& direction,
	float tMin,
	float tMax,
	HitInfo* hit)
{
	const vec3 oc = shape.center - origin;
	const float a = glm::dot(direction, direction);
	const float h = glm::dot(direction, oc);
	const float c = glm::dot(oc, oc) - shape.radius * shape.radius;

	const float discriminant = h * h - a * c;
	if (discriminant < 0.0f)
		return false;

	const float sqrtd = sqrtf(discriminant);

	// Find the nearest root that lies in the acceptable range.
	float root = (h - sqrtd) / a;
	if (root <= tMin || tMax <= root)
	{
		root = (h + sqrtd) / a;
		if (root <= tMin || tMax <= root)
			return false;
	}

	if (hit != nullptr)
	{
		hit->t = root;
		hit->p = rayAt(origin, direction, root);
		hit->normal = (hit->p - shape.center) / shape.radius;
		hit->materialIndex = shape.materialIndex;
		sphereUv(hit->normal, hit->u, hit->v);
	}

	return true;
}

__device__ bool intersectQuad(
	const CudaShape& shape,
	const vec3& origin,
	const vec3& direction,
	float tMin,
	float tMax,
	HitInfo* hit)
{
	const vec3 n = glm::cross(shape.u, shape.v);
	const vec3 normal = glm::normalize(n);
	const float denom = glm::dot(normal, direction);

	// No hit if the ray is parallel to the plane.
	if (fabsf(denom) < 1e-8f)
		return false;

	// Return false if the hit point parameter t is outside the ray interval.
	const float d = glm::dot(normal, shape.q);
	const float t = (d - glm::dot(normal, origin)) / denom;

	if (t < tMin || t > tMax)
		return false;

	// Determine if the hit point lies within the planar shape using its plane coordinates.
	const vec3 intersection = rayAt(origin, direction, t);
	const vec3 planarHit = intersection - shape.q;
	const vec3 w = n / glm::dot(n, n);

	const float alpha = glm::dot(w, glm::cross(planarHit, shape.v));
	const float beta = glm::dot(w, glm::cross(shape.u, planarHit));

	if ((alpha < 0.0f || alpha > 1.0f) || (beta < 0.0f || beta > 1.0f))
		return false;

	if (hit != nullptr)
	{
		hit->t = t;
		hit->p = intersection;
		hit->normal = normal;
		hit->materialIndex = shape.materialIndex;
		hit->u = alpha;
		hit->v = beta;
	}

	return true;
}

__device__ bool intersectShape(
	const CudaShape& shape,
	const vec3& origin,
	const vec3& direction,
	float tMin,
	float tMax,
	HitInfo* hit)
{
	if (shape.type == CUDA_SHAPE_SPHERE)
		return intersectSphere(shape, origin, direction, tMin, tMax, hit);

	return intersectQuad(shape, origin, direction, tMin, tMax, hit);
}

__device__ bool intersectScene(
	const CudaShape* shapes,
	int shapeCount,
	const vec3& origin,
	const vec3& direction,
	float tMin,
	float tMax,
	HitInfo* hit)
{
	HitInfo tempHit;
	bool hitAnything = false;
	float closest = tMax;

	for (int i = 0; i < shapeCount; ++i)
	{
		if (intersectShape(shapes[i], origin, direction, tMin, closest, hit != nullptr ? &tempHit : nullptr))
		{
			if (hit == nullptr)
				return true;

			hitAnything = true;
			closest = tempHit.t;
			*hit = tempHit;
		}
	}

	return hitAnything;
}

__device__ vec3 textureColor(const CudaTexture* textures, int textureIndex, float u, float v)
{
	for (int depth = 0; depth < 16 && textureIndex >= 0; ++depth)
	{
		const CudaTexture& texture = textures[textureIndex];
		if (texture.type == CUDA_TEXTURE_CONSTANT)
			return texture.color;

		float uc = u * texture.columns;
		float vf = v * texture.rows;
		int uInt = static_cast<int>(floorf(uc));
		int vInt = static_cast<int>(floorf(vf));
		float uLocal = uc - uInt;
		float vLocal = vf - vInt;
		const bool isEven = ((uInt + vInt) % 2) == 0;
		u = uLocal;
		v = vLocal;
		textureIndex = isEven ? texture.texture1 : texture.texture2;
	}

	return { 0.0f, 0.0f, 0.0f };
}

__device__ vec3 materialColor(
	const CudaMaterial* materials,
	const CudaTexture* textures,
	const HitInfo& hit)
{
	const CudaMaterial& material = materials[hit.materialIndex];
	if (material.textureIndex >= 0)
		return textureColor(textures, material.textureIndex, hit.u, hit.v);

	return material.color;
}

__device__ vec3 shadeLight(
	const CudaLight& light,
	const CudaMaterial* materials,
	const CudaTexture* textures,
	const vec3& rayDirection,
	const HitInfo& hit)
{
	// Color resultante
	vec3 ret = { 0.0f, 0.0f, 0.0f };

	vec3 _direction = light.type == CUDA_LIGHT_POINT
		                  ? light.position - hit.p // direccion desde la luz hasta el punto de colision
		                  : light.direction;

	_direction = glm::normalize(_direction);
	const vec3 normal = glm::normalize(hit.normal);
	const float lightIntensity = fmaxf(0.0f, glm::dot(normal, _direction));

	// -- difusa
	const vec3 diffuseLighting = light.color * lightIntensity;

	// -- especular
	const vec3 view = glm::normalize(-rayDirection);
	const vec3 halfVector = glm::normalize(_direction + view);
	float specular = fmaxf(0.0f, glm::dot(normal, halfVector));
	specular = powf(specular, static_cast<float>(light.glossPower));
	const vec3 specularColor = light.color * specular;

	ret = materialColor(materials, textures, hit) * diffuseLighting + specularColor;
	return ret;
}

__device__ vec3 rayColor(
	const CudaShape* shapes, int shapeCount,
	const CudaMaterial* materials,
	const CudaTexture* textures,
	const CudaLight* lights, int lightCount,
	vec3 origin, vec3 direction)
{
	vec3 accumulated = { 0.0f, 0.0f, 0.0f };
	vec3 throughput = { 1.0f, 1.0f, 1.0f };

	for (int depth = 0; depth < 10; ++depth)
	{
		HitInfo hit;
		if (!intersectScene(shapes, shapeCount, origin, direction, 0.001f, 100.0f, &hit))
			break;

		const CudaMaterial& material = materials[hit.materialIndex];
		vec3 local = { 0.0f, 0.0f, 0.0f };
		local += materialColor(materials, textures, hit) * vec3(0.1f, 0.1f, 0.1f); // Luz de ambiente cableada

		for (int i = 0; i < lightCount; ++i)
		{
			const CudaLight& light = lights[i];
			if (light.shadow)
			{
				// direccion desde el punto de colision hasta la luz
				vec3 shadowDir = light.position - hit.p;
				const float maxDistance = glm::length(shadowDir);
				shadowDir = glm::normalize(shadowDir);

				if (intersectScene(shapes, shapeCount, hit.p, shadowDir, 0.001f, maxDistance, nullptr))
					continue;
			}

				local = local + shadeLight(light, materials, textures, direction, hit);
		}

		accumulated = accumulated + throughput * local;

		if (material.glossFactor <= 0.0f)
			break;

		origin = hit.p + hit.normal * 0.001f;
		direction = glm::reflect(glm::normalize(direction), glm::normalize(hit.normal));
		throughput = throughput * material.glossFactor;
	}

	return accumulated;
}

__device__ unsigned char unitTo255(float v)
{
	if (v < 0.0f) return 0;
	if (v > 1.0f) return 255;
	return static_cast<unsigned char>(255.99f * v);
}

__global__ void renderKernel( // parametros
	unsigned char* pixels, int width, int height,
	const CudaCamera* camera,
	const CudaShape* shapes, int shapeCount,
	const CudaMaterial* materials,
	const CudaTexture* textures,
	const CudaLight* lights, int lightCount
)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height || x < 0 || y < 0)
		return;

	const vec3 sample = camera->positionTopLeft
		+ camera->deltaX * static_cast<float>(x)
		+ camera->deltaY * static_cast<float>(y);
	const vec3 direction = glm::normalize(sample - camera->position);
	const vec3 color = rayColor(
		shapes, shapeCount,
		materials, textures,
		lights, lightCount,
		camera->position,
		direction);

	const int idx = (y * width + x) * 4;
	pixels[idx + 0] = unitTo255(color.x);
	pixels[idx + 1] = unitTo255(color.y);
	pixels[idx + 2] = unitTo255(color.z);
	pixels[idx + 3] = 255;
}

template <typename T>
bool uploadArray(T** devicePtr, const std::vector<T>& hostData)
{
	if (hostData.empty())
		return true;

	if (cudaMalloc(reinterpret_cast<void**>(devicePtr), hostData.size() * sizeof(T)) != cudaSuccess)
		return false;

	return cudaMemcpy(*devicePtr, hostData.data(), hostData.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}

CUDABackend::CUDABackend(
	int width,
	int height,
	const CudaCamera& camera,
	const std::vector<CudaShape>& shapes,
	const std::vector<CudaMaterial>& materials,
	const std::vector<CudaTexture>& textures,
	const std::vector<CudaLight>& lights)
	: _width(width),
	  _height(height),
	  _shapeCount(static_cast<int>(shapes.size())),
	  _materialCount(static_cast<int>(materials.size())),
	  _textureCount(static_cast<int>(textures.size())),
	  _lightCount(static_cast<int>(lights.size()))
{
	if (cudaMalloc(reinterpret_cast<void**>(&_deviceCamera), sizeof(CudaCamera)) != cudaSuccess)
		return;

	if (cudaMemcpy(_deviceCamera, &camera, sizeof(CudaCamera), cudaMemcpyHostToDevice) != cudaSuccess)
		return;

	if (!uploadArray(&_deviceShapes, shapes))
		return;

	if (!uploadArray(&_deviceMaterials, materials))
		return;

	if (!uploadArray(&_deviceTextures, textures))
		return;

	if (!uploadArray(&_deviceLights, lights))
		return;

	if (cudaMalloc(reinterpret_cast<void**>(&_devicePixels), width * height * 4) != cudaSuccess)
		return;

	_valid = true;
}

CUDABackend::~CUDABackend()
{
	cudaFree(_deviceCamera);
	cudaFree(_deviceShapes);
	cudaFree(_deviceMaterials);
	cudaFree(_deviceTextures);
	cudaFree(_deviceLights);
	cudaFree(_devicePixels);
}

bool CUDABackend::Render(unsigned char* hostPixels)
{
	if (!_valid || hostPixels == nullptr)
		return false;

	constexpr int blockSize = 16;
	constexpr dim3 block(blockSize, blockSize);
	const dim3 grid((_width + block.x - 1) / block.x, (_height + block.y - 1) / block.y);

	renderKernel<<<grid, block>>>(
		_devicePixels,
		_width,
		_height,
		_deviceCamera,
		_deviceShapes,
		_shapeCount,
		_deviceMaterials,
		_deviceTextures,
		_deviceLights,
		_lightCount);

	if (cudaGetLastError() != cudaSuccess)
		return false;

	if (cudaDeviceSynchronize() != cudaSuccess)
		return false;

	return cudaMemcpy(hostPixels, _devicePixels, _width * _height * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
}

#pragma once
#include <memory>

#include "Material.h"
#include "Ray.hpp"
#include "Shape.h"

class Sphere : public Shape
{
public:
	Sphere(point3 center, float radius, const std::shared_ptr<Material>& material);

	point3 GetCenter() const { return _center; }
	float GetRadius() const { return _radius; }
	std::shared_ptr<Material> GetMaterial() const { return _material; }

protected:
	bool Intersect(const Ray& ray, float tMin, float tMax) const override;
	bool Intersect(const Ray& ray, float tMin, float tMax, InfoIntersection& info) const override;

private:
	static void get_sphere_uv(const point3& p, float& u, float& v);

	point3 _center;
	float _radius;
	std::shared_ptr<Material> _material;
};


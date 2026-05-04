#pragma once
#include <memory>

#include "Material.h"
#include "Ray.hpp"
#include "Shape.h"

class Quad : public Shape
{
public:
	Quad(point3 Q, glm::vec3 u, glm::vec3 v, const std::shared_ptr<Material>& material);

	bool Intersect(const Ray& ray, float tMin, float tMax) const override;
	bool Intersect(const Ray& ray, float tMin, float tMax, InfoIntersection& info) const override;
	point3 GetQ() const { return _Q; }
	glm::vec3 GetU() const { return _u; }
	glm::vec3 GetV() const { return _v; }
	std::shared_ptr<Material> GetMaterial() const { return _material; }
protected:

	virtual bool is_interior(float a, float b, InfoIntersection& info) const;
	virtual bool is_interior(float a, float b) const;

private:
	glm::vec3 _u;
	glm::vec3 _v;
	std::shared_ptr<Material> _material;
	point3 _Q;

	glm::vec3 w;
	glm::vec3 normal;
	float D; // dot entre la normal y Q

	// v
	// |
	// +-- u
};
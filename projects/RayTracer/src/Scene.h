#pragma once
#include <memory>
#include <vector>

#include "Shape.h"

class Scene : public Shape
{
public:
	Scene() = default;

	void Add(std::shared_ptr<Shape> sh);
	bool Intersect(const Ray& ray, float tMin, float tMax) const override;
	bool Intersect(const Ray& ray, float tMin, float tMax, InfoIntersection& info) const override;

private:
	std::vector<std::shared_ptr<Shape>> _sceneShapes;
};
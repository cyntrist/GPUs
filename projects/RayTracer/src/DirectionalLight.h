#pragma once
#include "Light.h"

class DirectionalLight : public Light
{
public:
	DirectionalLight(glm::vec3 direction, Color color);

	Color Shade(Ray r, InfoIntersection& hit) override;
	glm::vec3 GetDirection() const { return _direction; }
	Color GetColor() const { return _color; }

private:
	glm::vec3 _direction;
	Color _color;
};

#pragma once
#include "Light.h"

class PointLight : public Light
{
public:
	PointLight(glm::vec3 position, Color color);

	Color Shade(Ray r, InfoIntersection& hit) override;

	glm::vec3 ShadowDir(glm::vec3 position) override;

private:
	glm::vec3 _position;
	Color _color;
};
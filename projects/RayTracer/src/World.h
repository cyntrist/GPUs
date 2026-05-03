#pragma once
#include <memory>
#include <vector>

#include "Light.h"
#include "Shape.h"

class World
{
public:
	World(std::shared_ptr<Shape> scene);

	std::shared_ptr<Shape> GetScene() const
	{
		return _scene;
	}

	std::vector<std::shared_ptr<Light>> GetLights()
	{
		return _lights;
	}

	void AddLight(std::shared_ptr<Light> light);

private:
	std::shared_ptr<Shape> _scene;
	std::vector<std::shared_ptr<Light>> _lights;
};
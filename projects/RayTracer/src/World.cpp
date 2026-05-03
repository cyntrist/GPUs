#include "World.h"

World::World(std::shared_ptr<Shape> scene)
	: _scene(scene)
{

}

void World::AddLight(std::shared_ptr<Light> light)
{
	_lights.push_back(light);
}
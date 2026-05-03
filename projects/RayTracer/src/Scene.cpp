#include "Scene.h"

void Scene::Add(std::shared_ptr<Shape> sh)
{
	_sceneShapes.push_back(sh);
}

bool Scene::Intersect(const Ray& ray, float tMin, float tMax) const
{
    for (const auto& sh : _sceneShapes)
    {
        if (sh->Intersect(ray, tMin, tMax))
            return true;
    }
    return false;
}

bool Scene::Intersect(const Ray& ray, float tMin, float tMax, InfoIntersection& info) const
{
    InfoIntersection ii;
    bool hitAnything = false;
    float closestSoFar = tMax;

    for (const auto& sh : _sceneShapes)
    {
        if (sh->Intersect(ray, tMin, closestSoFar, ii))
        {
            hitAnything = true;
            closestSoFar = ii.t;   
            info = ii;             
        }
    }

    return hitAnything;
}

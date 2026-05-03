#define _USE_MATH_DEFINES
#include <cmath>

#include "Sphere.h"
#include "Ray.hpp"
#include "glm/geometric.hpp"

Sphere::Sphere(point3 center, float radius, const std::shared_ptr<Material>& material)
	: _center(center), _radius(radius), _material(material)
{
	
}

bool Sphere::Intersect(const Ray& ray, float tMin, float tMax) const
{
    glm::vec3 oc = _center - ray.Origin();
    auto a = glm::squared_length(ray.Direction()); // he "hecho" squared_length pero realmente es lo mismo que glm::dot(v, v)
    auto h = dot(ray.Direction(), oc);
    auto c = glm::squared_length(oc) - _radius * _radius;

    auto discriminant = h * h - a * c;
    if (discriminant < 0)
        return false;

    auto sqrtd = std::sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (h - sqrtd) / a;
    if (root <= tMin || tMax <= root) {
        root = (h + sqrtd) / a;
        if (root <= tMin || tMax <= root)
            return false;
    }

    return true;
}

bool Sphere::Intersect(const Ray& ray, float tMin, float tMax, InfoIntersection& info) const
{
    glm::vec3 oc = _center - ray.Origin();
    auto a = glm::squared_length(ray.Direction());
    auto h = dot(ray.Direction(), oc);
    auto c = glm::squared_length(oc) - _radius * _radius;

    auto discriminant = h * h - a * c;
    if (discriminant < 0)
        return false;

    auto sqrtd = std::sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (h - sqrtd) / a;

    if (root <= tMin || tMax <= root) 
    {
        root = (h + sqrtd) / a;

        if (root <= tMin || tMax <= root)
        {
            return false;
        }
    }

    info.t = root;
    info.p = ray.At(info.t);
    info.normal = (info.p - _center) / _radius;

    info.m = _material;

    get_sphere_uv(info.normal, info.u, info.v);

    return true;
}

void Sphere::get_sphere_uv(const point3& p, float& u, float& v)
{
    // p: a given point on the sphere of radius one, centered at the origin.
    float theta = std::acos(-p.y);
    float phi = std::atan2(-p.z, p.x) + M_PI;

    u = phi / (2 * M_PI);
    v = theta / M_PI;
}
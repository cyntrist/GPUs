#include "Quad.h"

#include "glm/geometric.hpp"

Quad::Quad(point3 Q, glm::vec3 u, glm::vec3 v, const std::shared_ptr<Material>& material)
	: _Q(Q), _u(u), _v(v), _material(material)
{


    auto n = glm::cross(_u, _v);
    normal = glm::normalize(n);

    D = dot(normal, Q);
    w = n / dot(n, n);
}

bool Quad::Intersect(const Ray& ray, float tMin, float tMax) const
{
    auto denom = glm::dot(normal, ray.Direction());

    // No hit if the ray is parallel to the plane.
    if (std::fabs(denom) < 1e-8)
        return false;

    // Return false if the hit point parameter t is outside the ray interval.
    auto t = (D - glm::dot(normal, ray.Origin())) / denom;

    if (t < tMin || t > tMax)
        return false;

    // Determine if the hit point lies within the planar shape using its plane coordinates.
    auto intersection = ray.At(t);
    glm::vec3 planar_hitpt_vector = intersection - _Q;

    auto alpha = glm::dot(w, cross(planar_hitpt_vector, _v));
    auto beta = glm::dot(w, cross(_u, planar_hitpt_vector));

    if (!is_interior(alpha, beta))
        return false;

    return true;
}

bool Quad::Intersect(const Ray& ray, float tMin, float tMax, InfoIntersection& info) const
{
    auto denom = glm::dot(normal, ray.Direction());

    // No hit if the ray is parallel to the plane.
    if (std::fabs(denom) < 1e-8)
        return false;

    // Return false if the hit point parameter t is outside the ray interval.
    auto t = (D - glm::dot(normal, ray.Origin())) / denom;

    if (t < tMin || t > tMax)
        return false;

    // Determine if the hit point lies within the planar shape using its plane coordinates.
    auto intersection = ray.At(t);
    glm::vec3 planar_hitpt_vector = intersection - _Q;

    auto alpha = glm::dot(w, cross(planar_hitpt_vector, _v));
    auto beta = glm::dot(w, cross(_u, planar_hitpt_vector));

    if (!is_interior(alpha, beta, info))
        return false;

    // Ray hits the 2D shape; set the rest of the hit record and return true.
    info.t = t;
    info.p = intersection;
    info.m = _material;
    info.normal = normal;

    return true;
}

bool Quad::is_interior(float a, float b, InfoIntersection& info) const
{
    // Given the hit point in plane coordinates, return false if it is outside the
    // primitive, otherwise set the hit record UV coordinates and return true.
    if ((a < 0 || a > 1) || (b < 0 || b > 1))
		return false;

    info.u = a;
    info.v = b;

    return true;
}

bool Quad::is_interior(float a, float b) const
{
    // Given the hit point in plane coordinates, return false if it is outside the
    // primitive, otherwise set the hit record UV coordinates and return true.
    if ((a < 0 || a > 1) || (b < 0 || b > 1))
        return false;

    return true;
}
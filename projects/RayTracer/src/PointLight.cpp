#include "PointLight.h"

#include "glm/geometric.hpp"

PointLight::PointLight(glm::vec3 position, Color color)
	: _position(position), _color(color)
{
    _shadow = true;
}

Color PointLight::Shade(Ray r, InfoIntersection& hit)
{
    Color ret; // color resultante

    //glm::vec3 _direction = hit.p - _position; // direccion desde la luz hasta el punto de colision
    glm::vec3 _direction = _position - hit.p; // direccion desde la luz hasta el punto de colision
    float _distance = glm::squared_length(_direction);
    _direction = glm::normalize(_direction);

    glm::vec3 normal = glm::normalize(hit.normal);
    float lightIntensity = glm::max(0.0f, glm::dot(normal, _direction));

    // -- difusa
    glm::vec3 diffuse = _color * lightIntensity;
    glm::vec3 diffuseLighting = diffuse;

    // -- especular
    glm::vec3 view = glm::normalize(-r.Direction());
    glm::vec3 halfVector = glm::normalize(_direction + view);
    float specular = std::max(0.0f, glm::dot(normal, halfVector));
    specular = std::pow(specular, _glossPower);
    glm::vec3 specularColor = _color * specular;

    ret = hit.m->GetColor(hit) * diffuseLighting + specularColor;

    return ret;
}

glm::vec3 PointLight::ShadowDir(glm::vec3 position)
{
    // direccion desde el punto de colision hasta la luz
    return _position - position;
}

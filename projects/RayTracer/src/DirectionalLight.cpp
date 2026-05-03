#include "DirectionalLight.h"

#include <iostream>
#include "glm/geometric.hpp"

DirectionalLight::DirectionalLight(glm::vec3 direction, Color color)
	: _direction(direction), _color(color)
{
    _shadow = false;
}

Color DirectionalLight::Shade(Ray r, InfoIntersection& hit)
{
    Color ret; // color resultante
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
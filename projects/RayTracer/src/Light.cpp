#include "Light.h"

Color Light::Shade(Ray r, InfoIntersection& hit)
{
	Color ret; // color resultante

	// Ambiente
	ret += WHITE; // Luz de ambiente cableada

	return ret; 
}

glm::vec3 Light::ShadowDir(glm::vec3 position)
{
	return glm::vec3();
}
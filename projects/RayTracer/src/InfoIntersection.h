#pragma once
#include "Material.h"
#include <memory>
struct InfoIntersection // almacena la informacion de una colision
{
	std::shared_ptr<Material> m; // el material del que esta hecho el objeto con el que se ha colisionado el rayo
	glm::vec3 p; // punto posicion de la interseccion
	glm::vec3 normal; // normal de la interseccion
	float t; // tiempo de la distancia en el que el rayo ha producido colision, solamente devuelve la info de la primera colision (para controlar objetos encima de objetos)

	float u;
	float v;
};
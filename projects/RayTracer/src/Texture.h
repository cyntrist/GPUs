#pragma once
#include "Color.h"

class Texture
{
public:
	virtual ~Texture() = default;
	virtual Color color(float u, float v) const = 0;
};

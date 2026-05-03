#pragma once
#include "Color.h"
#include "Texture.h"

class ConstantTexture : public Texture
{
public:
	ConstantTexture(Color color);

	Color color(float u, float v) const override;

private:
	Color _color;
};

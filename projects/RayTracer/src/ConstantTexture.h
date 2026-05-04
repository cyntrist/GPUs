#pragma once
#include "Color.h"
#include "Texture.h"

class ConstantTexture : public Texture
{
public:
	ConstantTexture(Color color);

	Color color(float u, float v) const override;
	Color GetStoredColor() const { return _color; }

private:
	Color _color;
};

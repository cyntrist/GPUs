#include "ConstantTexture.h"

ConstantTexture::ConstantTexture(Color color)
	: _color(color)
{
}

Color ConstantTexture::color(float u, float v) const
{
	return _color;
}
#pragma once
#include <memory>

#include "Texture.h"

class CheckerTexture : public Texture
{
public:
	CheckerTexture(const std::shared_ptr<Texture>& texture1, const std::shared_ptr<Texture>& texture2,
		int f, int c);

	Color color(float u, float v) const override;

private:
	std::shared_ptr<Texture> _texture1;
	std::shared_ptr<Texture> _texture2;

	int _f; // filas
	int _c; // columnas
};
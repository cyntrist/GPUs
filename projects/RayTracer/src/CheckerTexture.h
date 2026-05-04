#pragma once
#include <memory>

#include "Texture.h"

class CheckerTexture : public Texture
{
public:
	CheckerTexture(const std::shared_ptr<Texture>& texture1, const std::shared_ptr<Texture>& texture2,
		int f, int c);

	Color color(float u, float v) const override;
	std::shared_ptr<Texture> GetTexture1() const { return _texture1; }
	std::shared_ptr<Texture> GetTexture2() const { return _texture2; }
	int GetRows() const { return _f; }
	int GetColumns() const { return _c; }

private:
	std::shared_ptr<Texture> _texture1;
	std::shared_ptr<Texture> _texture2;

	int _f; // filas
	int _c; // columnas
};
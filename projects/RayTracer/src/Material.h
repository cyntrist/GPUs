#pragma once
#include <memory>
#include "Color.h"

class InfoIntersection;
class Texture;

class Material
{
public: 
	Material(Color color, const std::shared_ptr<Texture>& texture = nullptr);

	Color GetColor(InfoIntersection& ii) const;

	float GetGlossFactor() const
	{
		return _glossFactor;
	}

	void SetGlossFactor(float gf)
	{
		_glossFactor = gf;
	}

private:
	Color _color;
	float _glossFactor = 0;

	std::shared_ptr<Texture> _texture;
};

#include "Material.h"
#include "Texture.h"
#include "InfoIntersection.h"

Material::Material(Color color, const std::shared_ptr<Texture>& texture)
	: _color(color), _texture(texture)
{

}

Color Material::GetColor(InfoIntersection& ii) const
{
	if (_texture != nullptr)
		return _texture->color(ii.u, ii.v);
	return _color; // si es nula llama al GetColor por defecto
}

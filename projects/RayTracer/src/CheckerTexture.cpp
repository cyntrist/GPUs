#include "CheckerTexture.h"
#include <cmath>

CheckerTexture::CheckerTexture(const std::shared_ptr<Texture>& texture1, const std::shared_ptr<Texture>& texture2, 
    int f, int c)
	    : _texture1(texture1), _texture2(texture2), _f(f), _c(c)
{
}

Color CheckerTexture::color(float u, float v) const
{
    float uc = u * _c;
    float vf = v * _f;
    int uInt = std::floor(uc);
    int vInt = std::floor(vf);
    float uLocal = uc - uInt;
    float vLocal = vf - vInt;

    bool isEven = (uInt + vInt) % 2 == 0;

    return isEven ? _texture1->color(uLocal, vLocal) : _texture2->color(uLocal, vLocal);
}

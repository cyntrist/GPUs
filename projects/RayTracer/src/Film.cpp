#include "Film.h"

Film::Film(int x, int y, std::ostream& output)
    : _tamX(x), _tamY(y), _aspectRatio(1.0f * _tamX / _tamY), _out(output)
{
}

Film::Film(int x, int y)
    : _tamX(x), _tamY(y), _aspectRatio(1.0f * _tamX / _tamY)
{
    //_pixels.reserve(_tamX * _tamY * 4);
    _pixels.resize(_tamX * _tamY * 4, 0);
}

// original
void Film::AddPixel(Color color)
{
    if (missingHeader)
    {
        _out << "P3\n" << _tamX << ' ' << _tamY << "\n255\n";
        missingHeader = false;
    }

    int ir = (int)(255.99 * color.x);
    int ig = (int)(255.99 * color.y);
    int ib = (int)(255.99 * color.z);

    _out << ir << ' ' << ig << ' ' << ib << '\n';
}

unsigned char Film::UnitTo255(float v) 
{
    if (v < 0.0f) return 0;
    if (v > 1.0f) return 255;
    return static_cast<unsigned char>(255.99f * v);
}

//void Film::AddPixel(Color color)
//{ // esto esta MAL!!!
//    _pixels.push_back(UnitTo255(color.x));
//    _pixels.push_back(UnitTo255(color.y));
//    _pixels.push_back(UnitTo255(color.z));
//    _pixels.push_back(255);
//}

void Film::AddPixel(int x, int y, Color color)
{
    int idx = (y * _tamX + x) * 4;

    _pixels[idx + 0] = UnitTo255(color.x);
    _pixels[idx + 1] = UnitTo255(color.y);
    _pixels[idx + 2] = UnitTo255(color.z);
    _pixels[idx + 3] = 255;
}
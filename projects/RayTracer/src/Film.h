#ifndef _FILM_H
#define _FILM_H
#include "Color.h"
#include <iostream>
#include <vector>

class Film {
public:
    Film(int x, int y, std::ostream& output);
    Film(int x, int y);

    static unsigned char UnitTo255(float v);
    void AddPixel(Color color);
    void AddPixel(int x, int y, Color color);

    int GetTamX() const { return _tamX; }
    int GetTamY() const { return _tamY; }
    float GetAspectRatio() const { return _aspectRatio; }
    const unsigned char* Data() const { return _pixels.data(); }
    unsigned char* MutableData() { return _pixels.data(); }
    int Pitch() const { return _tamX * 4; } // numero de bytes que ocupa una fila completa de pixeles en memoria -> anchura * 4 (bytes por pixel
    // SDL_render.h:  the number of bytes in a row of pixel data, including padding*between lines.

private:
    int _tamX;
    int _tamY;
    float _aspectRatio;
	std::ostream& _out = std::cout;
    bool missingHeader = true;
    std::vector<unsigned char> _pixels;
};

#endif
#include "Renderer.h"

#include "glm/geometric.hpp"

Renderer::Renderer(std::shared_ptr<Film> film, std::shared_ptr<Camera> camera, std::shared_ptr<World> world)
	: _film(film), _camera(camera), _world(world)
{

}

Renderer::~Renderer()
{
}

void Renderer::Render()
{
    for (int y = 0; y < _film->GetTamY(); ++y)
    {
        for (int x = 0; x < _film->GetTamX(); ++x)
        {
            //std::cout << "Escribiendo pixel numero " << x << " " << y << " de ancho " << _film->GetTamX() << " e y " << _film->GetTamY() << ".\n";
            const Ray ray_primary = _camera->GetRay(x, y);
            const Color c = RayColor(ray_primary, 10);
            _film->AddPixel(x, y, c);
        }
    }
}

Color Renderer::RayColor(const Ray& r, int k)
{
    if (k <= 0) return BLACK;

    InfoIntersection ii;
    if (_world->GetScene()->Intersect(r, 0.001f, 100, ii))
    {
        Color color = BLACK;
        color += (ii.m->GetColor(ii) * Color(0.1, 0.1, 0.1)); // Luz de ambiente cableada

        for (auto& l : _world->GetLights())
        {
            if (l->GetShadow())
            {
                glm::vec3 origin = ii.p;
                glm::vec3 dir = l->ShadowDir(ii.p);
                float maxDist = glm::length(dir);
                dir = glm::normalize(dir);

                Ray shadowRay(origin, dir);

                if (_world->GetScene()->Intersect(shadowRay, 0.001f, maxDist))
                {
	                // saltar a la siguiente geometria
                    continue;
                }
            }
            color += l->Shade(r, ii);
        }
        
        if (ii.m->GetGlossFactor() > 0.0f && k > 0)
        {
            glm::vec3 hitPos = ii.p + ii.normal * 0.001f;
            const glm::vec3 normal = glm::normalize(ii.normal);
            const glm::vec3 dir = glm::reflect(glm::normalize(r.Direction()), normal);

            Ray shadowRay(hitPos, dir);

            color += ii.m->GetGlossFactor() * RayColor(shadowRay, k - 1);
        }

        return color;
    }

    // Skybox
    return BLACK;
}
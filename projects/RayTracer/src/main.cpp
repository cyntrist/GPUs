// Nieves Alonso Gilsanz
// Cynthia Trist�n �lvarez
// Grupo 01
// Iluminaci�n y Materiales
// https://raytracing.github.io/books/RayTracingInOneWeekend.html

#include "glm/vec3.hpp"
#include "glm/geometric.hpp"
#include "Film.h"
#include "Color.h"
#include "Camera.hpp"
#include <memory>
#include <fstream>

#include "CheckerTexture.h"
#include "ConstantTexture.h"
#include "DirectionalLight.h"
#include "PointLight.h"
#include "Quad.h"
#include "SDLViewer.h"
#include "Renderer.h"
#include "Scene.h"
#include "Sphere.h"

#include "checkML.h"
#include "checkMLNew.h"

int main(void)
{
    // pelicula
    auto film = std::make_shared<Film>(1920, 1080);

    glm::vec3 position = { 0.0f, 0.0f, 3.0f };
    glm::vec3 look = { 0.0f, 0.0f, -1.0f };
    glm::vec3 up = { 0.0f, 1.0f, 0.0f };

    // camara
    auto cam = std::make_shared<Camera>(position, look, up, film, 60.0f);

    // materiales basicos
    auto azul = std::make_shared<Material>(BLUE);
    auto amarillo = std::make_shared<Material>(YELLOW);
    auto rojo = std::make_shared<Material>(RED);
    auto verde = std::make_shared<Material>(GREEN);

    // gloss factor
    azul->SetGlossFactor(0.5f);
    amarillo->SetGlossFactor(0.5f);
    rojo->SetGlossFactor(0.5f);
    verde->SetGlossFactor(0.5f);

    // texturas
    auto verde_textura = std::make_shared<ConstantTexture>(GREEN);
    auto blanco_textura = std::make_shared<ConstantTexture>(WHITE);
    auto magenta_textura = std::make_shared<ConstantTexture>(MAGENTA);
    auto negro_textura = std::make_shared<ConstantTexture>(BLACK);
    auto ajedrez = std::make_shared<CheckerTexture>(verde_textura, blanco_textura, 10, 10);
    auto imaginacion = std::make_shared<CheckerTexture>(negro_textura, magenta_textura, 10, 10);
    auto recursion = std::make_shared<CheckerTexture>(blanco_textura, imaginacion, 10, 10);

    // material con texturas
    auto ajedrez_material = std::make_shared<Material>(BLACK, ajedrez);
    auto imaginacion_material = std::make_shared<Material>(BLACK, imaginacion);
    auto recursion_material = std::make_shared<Material>(BLACK, recursion);

    // objetos
    auto obj5 = std::make_shared<Quad>(point3(3, -1, -5), glm::vec3(-6, 0, 0), glm::vec3(0, 0, 6), recursion_material);
    auto obj4 = std::make_shared<Sphere>(glm::vec3(0, -100, -2), 99.0f, verde);
    auto obj3 = std::make_shared<Sphere>(glm::vec3(-2, 0, -2), 1.0f, rojo);
    auto obj2 = std::make_shared<Sphere>(glm::vec3(0, 0, -2), 1.0f, amarillo);
    auto obj1 = std::make_shared<Sphere>(glm::vec3(2, 0, -2), 1.0f, azul);

    // luces
    glm::vec3 dir = { 1.0f, 1.0f, 0.0f };
    auto dirLight = std::make_shared<DirectionalLight>(dir, WHITE);

    glm::vec3 pos = { 2.0f, 2.0f, 0.0f };
    auto pointLight = std::make_shared<PointLight>(pos, WHITE);
    glm::vec3 pos2 = { -2.0f, 2.0f, -3.0f };
    auto pointLight2 = std::make_shared<PointLight>(pos2, WHITE);

    dirLight->SetGlossPower(400);
    pointLight->SetGlossPower(400);
    pointLight2->SetGlossPower(400);

    // mundo
    auto scene = std::make_shared<Scene>();
    scene->Add(obj1);
    scene->Add(obj2);
    scene->Add(obj3);
    scene->Add(obj5);
    auto world = std::make_shared<World>(scene);

    //world->AddLight(dirLight);
    world->AddLight(pointLight);
    world->AddLight(pointLight2);

    // renderer
    Renderer renderer(film, cam, world);

    // sdl
    SDLViewer viewer(film, &renderer);
    viewer.Loop();

    return 0;
}

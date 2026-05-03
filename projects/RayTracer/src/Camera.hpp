#pragma once

#include <memory>
#include "glm/vec3.hpp"
#include "Film.h"
#include "Ray.hpp"

class Camera 
{
public:
    Camera(
        glm::vec3 position,
        glm::vec3 look,
        glm::vec3 up,
        const std::shared_ptr<Film>& film,
        float fov_degrees_vertical);

    Ray GetRay(int x, int y) const;

private:
    glm::vec3 position;
    glm::vec3 delta_x;
    glm::vec3 delta_y;
    glm::vec3 position_top_left;
};
#include "Camera.hpp"
#include <cmath>
#include "glm/geometric.hpp"
#include "glm/trigonometric.hpp"

Camera::Camera(
    glm::vec3 position,
    glm::vec3 look,
    glm::vec3 up,
    const std::shared_ptr<Film>& film,
    float fov_degrees_vertical
) : position(position) 
{
    const float fov_radians_vertical = glm::radians(fov_degrees_vertical * 0.5);
    const float half_height_normalized = std::tan(fov_radians_vertical);

    const glm::vec3 forward_displacement = position - look;
    const float focal_length = glm::length(forward_displacement);
    const glm::vec3 forward = forward_displacement / focal_length;
    const glm::vec3 right = glm::cross(up, forward);    
    const glm::vec3 v = glm::cross(forward, right);

    const float half_height_viewport = focal_length * half_height_normalized;
    const float half_width_viewport = half_height_viewport * film->GetAspectRatio();

    const float height_viewport = half_height_viewport * 2.0;
    const float width_viewport = half_width_viewport * 2.0;

    const float pixel_height = height_viewport / float(film->GetTamY());
    const float pixel_width = width_viewport / float(film->GetTamX());

    delta_x = right * pixel_width;
    delta_y = -v * pixel_height;

    position_top_left =
        position - focal_length * forward
        + v * half_height_viewport + delta_x * 0.5f
        - right * half_width_viewport + delta_y * 0.5f;
}

Ray Camera::GetRay(int x, int y) const 
{
    const glm::vec3 sample = position_top_left + delta_x * (float)x + delta_y * (float)y;
    const glm::vec3 displacement = (sample - position);

    return Ray{position, glm::normalize(displacement)};
}
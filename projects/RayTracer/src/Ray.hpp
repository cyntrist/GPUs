#ifndef RAY_H
#define RAY_H

#include "glm/vec3.hpp"

using point3 = glm::vec3;

class Ray {
  public:
    Ray() {}

    Ray(const point3& origin, const glm::vec3& direction) : orig(origin), dir(direction) {}

    const point3& Origin() const  { return orig; }
    const glm::vec3& Direction() const { return dir; }

    point3 At(float t) const {
        return orig + t*dir;
    }

  private:
    point3 orig;
    glm::vec3 dir;
};

#endif
/*
    This file is part of the example codes which have been used
    for the "Code Optmization Workshop".
    
    Copyright (C) 2016  Fabio Baruffa <fbaru-dev@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _PARTICLE_HPP
#define _PARTICLE_HPP
#include <cmath>
#include <new>
#include "types.hpp"
#include <sycl/sycl.hpp>
using namespace sycl;

struct ParticleAoS
{
  public:
    ParticleAoS() { init();}
    void init() 
    {
      pos[0] = 0.; pos[1] = 0.; pos[2] = 0.;
      vel[0] = 0.; vel[1] = 0.; vel[2] = 0.;
      acc[0] = 0.; acc[1] = 0.; acc[2] = 0.;
      mass   = 0.;
    }
    real_type pos[3];
    real_type vel[3];
    real_type acc[3];  
    real_type mass;
};

struct ParticleSoA
{
  public:
    ParticleSoA()
      : pos_x(nullptr), pos_y(nullptr), pos_z(nullptr),
        vel_x(nullptr), vel_y(nullptr), vel_z(nullptr),
        acc_x(nullptr), acc_y(nullptr), acc_z(nullptr),
        mass(nullptr), _Q()
    {}
    ~ParticleSoA() {
      if (pos_x) { sycl::free(pos_x, _Q); pos_x = nullptr; }
      if (pos_y) { sycl::free(pos_y, _Q); pos_y = nullptr; }
      if (pos_z) { sycl::free(pos_z, _Q); pos_z = nullptr; }
      if (vel_x) { sycl::free(vel_x, _Q); vel_x = nullptr; }
      if (vel_y) { sycl::free(vel_y, _Q); vel_y = nullptr; }
      if (vel_z) { sycl::free(vel_z, _Q); vel_z = nullptr; }
      if (acc_x) { sycl::free(acc_x, _Q); acc_x = nullptr; }
      if (acc_y) { sycl::free(acc_y, _Q); acc_y = nullptr; }
      if (acc_z) { sycl::free(acc_z, _Q); acc_z = nullptr; }
      if (mass)  { sycl::free(mass, _Q);  mass = nullptr; }
    }
    void init(int n, sycl::queue Q) 
    {
      _Q = Q;
      pos_x = malloc_shared<real_type>(n, Q); pos_y = malloc_shared<real_type>(n, Q); pos_z = malloc_shared<real_type>(n, Q);
      vel_x = malloc_shared<real_type>(n, Q); vel_y = malloc_shared<real_type>(n, Q); vel_z = malloc_shared<real_type>(n, Q);
      acc_x = malloc_shared<real_type>(n, Q); acc_y = malloc_shared<real_type>(n, Q); acc_z = malloc_shared<real_type>(n, Q);
      mass  = malloc_shared<real_type>(n, Q);
      if (!pos_x || !pos_y || !pos_z || !vel_x || !vel_y || !vel_z ||
          !acc_x || !acc_y || !acc_z || !mass)
      {
        this->~ParticleSoA();
        throw std::bad_alloc();
      }
    }
    real_type *pos_x, *pos_y, *pos_z;
    real_type *vel_x, *vel_y, *vel_z;
    real_type *acc_x, *acc_y, *acc_z;  
    real_type *mass;

    sycl::queue _Q;
};

#endif

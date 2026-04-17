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

#include "GSimulation.hpp"
#include "cpu_time.hpp"

#include <sycl/sycl.hpp>

#define SoA
#define GPU

GSimulation ::GSimulation() {
  std::cout << "===============================" << std::endl;
  std::cout << " Initialize Gravity Simulation" << std::endl;
  set_npart(16000);
  set_nsteps(10);
  set_tstep(0.1);
  set_sfreq(1);
}

GSimulation ::~GSimulation() {
#ifdef AoS
  delete particles;
#elif defined SoA
  delete particlesSoA;
#endif
}

void GSimulation ::init(int n) {
  // allocate particles
#ifdef AoS
  particles = new ParticleAoS[n];
#elif defined SoA
  particlesSoA = new ParticleSoA(n);
#endif
  init_pos();
  init_vel();
  init_acc();
  init_mass();
}

void GSimulation ::set_number_of_particles(int N) { set_npart(N); }

void GSimulation ::set_number_of_steps(int N) { set_nsteps(N); }

void GSimulation ::init_pos() {
  std::random_device rd; // random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(0, 1.0);

  for (int i = 0; i < get_npart(); ++i) {
#ifdef AoS
    particles[i].pos[0] = unif_d(gen);
    particles[i].pos[1] = unif_d(gen);
    particles[i].pos[2] = unif_d(gen);
#elif defined SoA
    (*particlesSoA).pos_x[i] = unif_d(gen);
    (*particlesSoA).pos_y[i] = unif_d(gen);
    (*particlesSoA).pos_z[i] = unif_d(gen);
#endif
  }
}

void GSimulation ::init_vel() {
  std::random_device rd; // random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(-1.0, 1.0);

  for (int i = 0; i < get_npart(); ++i) {
#ifdef AoS
    particles[i].vel[0] = unif_d(gen) * 1.0e-3f;
    particles[i].vel[1] = unif_d(gen) * 1.0e-3f;
    particles[i].vel[2] = unif_d(gen) * 1.0e-3f;
#elif defined SoA
    (*particlesSoA).vel_x[i] = unif_d(gen) * 1.0e-3f;
    (*particlesSoA).vel_y[i] = unif_d(gen) * 1.0e-3f;
    (*particlesSoA).vel_z[i] = unif_d(gen) * 1.0e-3f;
#endif
  }
}

void GSimulation ::init_acc() {
  for (int i = 0; i < get_npart(); ++i) {
#ifdef AoS
    particles[i].acc[0] = 0.f;
    particles[i].acc[1] = 0.f;
    particles[i].acc[2] = 0.f;
#elif defined SoA
    (*particlesSoA).acc_x[i] = 0.f;
    (*particlesSoA).acc_y[i] = 0.f;
    (*particlesSoA).acc_z[i] = 0.f;
#endif
  }
}

void GSimulation ::init_mass() {
  real_type n = static_cast<real_type>(get_npart());
  std::random_device rd; // random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(0.0, 1.0);

  for (int i = 0; i < get_npart(); ++i) {
#ifdef AoS
    particles[i].mass = n * unif_d(gen);
#elif defined SoA
    (*particlesSoA).mass[i] = n * unif_d(gen);
#endif
  }
}

void GSimulation ::get_acceleration(int n) {
  int i, j;

  const float softeningSquared = 1e-3f;
  const float G = 6.67259e-11f;

  for (i = 0; i < n; i++) // update acceleration
  {
#ifdef AoS
    real_type ax_i = particles[i].acc[0];
    real_type ay_i = particles[i].acc[1];
    real_type az_i = particles[i].acc[2];
#elif defined SoA
    real_type ax_i = (*particlesSoA).acc_x[i];
    real_type ay_i = (*particlesSoA).acc_y[i];
    real_type az_i = (*particlesSoA).acc_z[i];
#endif
    for (j = 0; j < n; j++) {
      real_type dx, dy, dz;
      real_type distanceSqr = 0.0f;
      real_type distanceInv = 0.0f;
#ifdef AoS
      dx = particles[j].pos[0] - particles[i].pos[0]; // 1flop
      dy = particles[j].pos[1] - particles[i].pos[1]; // 1flop
      dz = particles[j].pos[2] - particles[i].pos[2]; // 1flop
#elif defined SoA
      dx = (*particlesSoA).pos_x[j] - (*particlesSoA).pos_x[i]; // 1flop
      dy = (*particlesSoA).pos_y[j] - (*particlesSoA).pos_y[i]; // 1flop
      dz = (*particlesSoA).pos_z[j] - (*particlesSoA).pos_z[i]; // 1flop
#endif

      distanceSqr = dx * dx + dy * dy + dz * dz + softeningSquared; // 6flops
      distanceInv = 1.0f / sqrtf(distanceSqr); // 1div+1sqrt
#ifdef AoS
      ax_i += dx * G * particles[j].mass * distanceInv * distanceInv *
              distanceInv; // 6flops
      ay_i += dy * G * particles[j].mass * distanceInv * distanceInv *
              distanceInv; // 6flops
      az_i += dz * G * particles[j].mass * distanceInv * distanceInv *
              distanceInv; // 6flops
#elif defined SoA
      ax_i += dx * G * (*particlesSoA).mass[j] * distanceInv * distanceInv *
              distanceInv; // 6flops
      ay_i += dy * G * (*particlesSoA).mass[j] * distanceInv * distanceInv *
              distanceInv; // 6flops
      az_i += dz * G * (*particlesSoA).mass[j] * distanceInv * distanceInv *
              distanceInv; // 6flops
#endif
    }

#ifdef AoS
    particles[i].acc[0] = ax_i;
    particles[i].acc[1] = ay_i;
    particles[i].acc[2] = az_i;
#elif defined SoA
    (*particlesSoA).acc_x[i] = ax_i;
    (*particlesSoA).acc_y[i] = ay_i;
    (*particlesSoA).acc_z[i] = az_i;
#endif
  }
}

#ifdef GPU
void GSimulation ::get_acceleration_kernel(int n) {
  const float softeningSquared = 1e-3f;
  const float G = 6.67259e-11f;

  sycl::queue q;

#ifdef AoS
  sycl::buffer<real_type, 1> pos_x_buf(&particles->pos[0], sycl::range<1>(n));
  sycl::buffer<real_type, 1> pos_y_buf(&particles->pos[1], sycl::range<1>(n));
  sycl::buffer<real_type, 1> pos_z_buf(&particles->pos[2], sycl::range<1>(n));
  sycl::buffer<real_type, 1> acc_x_buf(&particles->acc[0], sycl::range<1>(n));
  sycl::buffer<real_type, 1> acc_y_buf(&particles->acc[1], sycl::range<1>(n));
  sycl::buffer<real_type, 1> acc_z_buf(&particles->acc[2], sycl::range<1>(n));
  sycl::buffer<real_type, 1> mass_buf (&particles->mass  , sycl::range<1>(n));
#elif defined SoA
  sycl::buffer<real_type, 1> pos_x_buf(particlesSoA->pos_x, sycl::range<1>(n));
  sycl::buffer<real_type, 1> pos_y_buf(particlesSoA->pos_y, sycl::range<1>(n));
  sycl::buffer<real_type, 1> pos_z_buf(particlesSoA->pos_z, sycl::range<1>(n));
  sycl::buffer<real_type, 1> acc_x_buf(particlesSoA->acc_x, sycl::range<1>(n));
  sycl::buffer<real_type, 1> acc_y_buf(particlesSoA->acc_y, sycl::range<1>(n));
  sycl::buffer<real_type, 1> acc_z_buf(particlesSoA->acc_z, sycl::range<1>(n));
  sycl::buffer<real_type, 1> mass_buf (particlesSoA->mass , sycl::range<1>(n));
#endif

  q.submit([&](sycl::handler &h) {
    auto pos_x = pos_x_buf.get_access<sycl::access::mode::read>(h);
    auto pos_y = pos_y_buf.get_access<sycl::access::mode::read>(h);
    auto pos_z = pos_z_buf.get_access<sycl::access::mode::read>(h);
    auto acc_x = acc_x_buf.get_access<sycl::access::mode::write>(h);
    auto acc_y = acc_y_buf.get_access<sycl::access::mode::write>(h);
    auto acc_z = acc_z_buf.get_access<sycl::access::mode::write>(h);
    auto mass  = mass_buf .get_access<sycl::access::mode::read>(h);

    h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { // update acceleration
      // acceleration always initializes to 0, there is no need to access particle value
      real_type ax_i = 0.0f;
      real_type ay_i = 0.0f;
      real_type az_i = 0.0f;

      for (int j = 0; j < n; j++) {
        real_type dx, dy, dz;
        real_type distanceSqr = 0.0f;
        real_type distanceInv = 0.0f;
        real_type factor;

        dx = pos_x[j] - pos_x[i]; // 1flop
        dy = pos_y[j] - pos_y[i]; // 1flop
        dz = pos_z[j] - pos_z[i]; // 1flop

        distanceSqr = dx * dx + dy * dy + dz * dz + softeningSquared; // 6flops
        distanceInv = 1.0f / sycl::sqrt(distanceSqr); // 1div+1sqrt

        factor = G * mass[j] * distanceInv * distanceInv * distanceInv;

        ax_i += dx * factor;
        ay_i += dy * factor;
        az_i += dz * factor;
      }

      acc_x[i] = ax_i; // 6flops
      acc_y[i] = ay_i; // 6flops
      acc_z[i] = az_i; // 6flops
    });
  }).wait();
}
#endif

real_type GSimulation ::updateParticles(int n, real_type dt) {
  int i;
  real_type energy = 0;

  for (i = 0; i < n; ++i) // update position
  {
#ifdef AoS
    particles[i].vel[0] += particles[i].acc[0] * dt; // 2flops
    particles[i].vel[1] += particles[i].acc[1] * dt; // 2flops
    particles[i].vel[2] += particles[i].acc[2] * dt; // 2flops

    particles[i].pos[0] += particles[i].vel[0] * dt; // 2flops
    particles[i].pos[1] += particles[i].vel[1] * dt; // 2flops
    particles[i].pos[2] += particles[i].vel[2] * dt; // 2flops

    particles[i].acc[0] = 0.;
    particles[i].acc[1] = 0.;
    particles[i].acc[2] = 0.;

    energy += particles[i].mass *
              (particles[i].vel[0] * particles[i].vel[0] +
               particles[i].vel[1] * particles[i].vel[1] +
               particles[i].vel[2] * particles[i].vel[2]); // 7flops
#elif defined SoA
    (*particlesSoA).vel_x[i] += (*particlesSoA).acc_x[i] * dt; // 2flops
    (*particlesSoA).vel_y[i] += (*particlesSoA).acc_y[i] * dt; // 2flops
    (*particlesSoA).vel_z[i] += (*particlesSoA).acc_z[i] * dt; // 2flops

    (*particlesSoA).pos_x[i] += (*particlesSoA).vel_x[i] * dt; // 2flops
    (*particlesSoA).pos_y[i] += (*particlesSoA).vel_y[i] * dt; // 2flops
    (*particlesSoA).pos_z[i] += (*particlesSoA).vel_z[i] * dt; // 2flops

    (*particlesSoA).acc_x[i] = 0.;
    (*particlesSoA).acc_y[i] = 0.;
    (*particlesSoA).acc_z[i] = 0.;

    energy += (*particlesSoA).mass[i] *
              ((*particlesSoA).vel_x[i] * (*particlesSoA).vel_x[i] +
               (*particlesSoA).vel_y[i] * (*particlesSoA).vel_y[i] +
               (*particlesSoA).vel_z[i] * (*particlesSoA).vel_z[i]); // 7flops
#endif
  }
  return energy;
}

#ifdef GPU
real_type GSimulation::updateParticles_kernel(int n, real_type dt) {
  real_type energy = 0;

  sycl::queue q;

#ifdef AoS
  sycl::buffer<real_type, 1> pos_x_buf(&particles->pos[0], sycl::range<1>(n));
  sycl::buffer<real_type, 1> pos_y_buf(&particles->pos[1], sycl::range<1>(n));
  sycl::buffer<real_type, 1> pos_z_buf(&particles->pos[2], sycl::range<1>(n));
  sycl::buffer<real_type, 1> vel_x_buf(&particles->vel[0], sycl::range<1>(n));
  sycl::buffer<real_type, 1> vel_y_buf(&particles->vel[1], sycl::range<1>(n));
  sycl::buffer<real_type, 1> vel_z_buf(&particles->vel[2], sycl::range<1>(n));
  sycl::buffer<real_type, 1> acc_x_buf(&particles->acc[0], sycl::range<1>(n));
  sycl::buffer<real_type, 1> acc_y_buf(&particles->acc[1], sycl::range<1>(n));
  sycl::buffer<real_type, 1> acc_z_buf(&particles->acc[2], sycl::range<1>(n));
  sycl::buffer<real_type, 1> mass_buf (&particles->mass,   sycl::range<1>(n));
  sycl::buffer<real_type, 1> energy_buf(&energy,          sycl::range<1>(1));
#elif defined SoA
  sycl::buffer<real_type, 1> pos_x_buf(particlesSoA->pos_x, sycl::range<1>(n));
  sycl::buffer<real_type, 1> pos_y_buf(particlesSoA->pos_y, sycl::range<1>(n));
  sycl::buffer<real_type, 1> pos_z_buf(particlesSoA->pos_z, sycl::range<1>(n));
  sycl::buffer<real_type, 1> vel_x_buf(particlesSoA->vel_x, sycl::range<1>(n));
  sycl::buffer<real_type, 1> vel_y_buf(particlesSoA->vel_y, sycl::range<1>(n));
  sycl::buffer<real_type, 1> vel_z_buf(particlesSoA->vel_z, sycl::range<1>(n));
  sycl::buffer<real_type, 1> acc_x_buf(particlesSoA->acc_x, sycl::range<1>(n));
  sycl::buffer<real_type, 1> acc_y_buf(particlesSoA->acc_y, sycl::range<1>(n));
  sycl::buffer<real_type, 1> acc_z_buf(particlesSoA->acc_z, sycl::range<1>(n));
  sycl::buffer<real_type, 1> mass_buf (particlesSoA->mass,  sycl::range<1>(n));
  sycl::buffer<real_type, 1> energy_buf(&energy          ,  sycl::range<1>(1));
#endif

  q.submit([&](sycl::handler &h) { // update position
    auto pos_x = pos_x_buf.get_access<sycl::access::mode::read_write>(h);
    auto pos_y = pos_y_buf.get_access<sycl::access::mode::read_write>(h);
    auto pos_z = pos_z_buf.get_access<sycl::access::mode::read_write>(h);
    auto vel_x = vel_x_buf.get_access<sycl::access::mode::read_write>(h);
    auto vel_y = vel_y_buf.get_access<sycl::access::mode::read_write>(h);
    auto vel_z = vel_z_buf.get_access<sycl::access::mode::read_write>(h);
    auto acc_x = acc_x_buf.get_access<sycl::access::mode::read_write>(h);
    auto acc_y = acc_y_buf.get_access<sycl::access::mode::read_write>(h);
    auto acc_z = acc_z_buf.get_access<sycl::access::mode::read_write>(h);
    auto mass  = mass_buf .get_access<sycl::access::mode::read>(h);
    auto energy_acc = energy_buf.get_access<sycl::access::mode::read_write>(h);

    h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
      vel_x[i] += acc_x[i] * dt; // 2flops
      vel_y[i] += acc_y[i] * dt; // 2flops
      vel_z[i] += acc_z[i] * dt; // 2flops

      pos_x[i] += vel_x[i] * dt; // 2flops
      pos_y[i] += vel_y[i] * dt; // 2flops
      pos_z[i] += vel_z[i] * dt; // 2flops

      acc_x[i] = 0.;
      acc_y[i] = 0.;
      acc_z[i] = 0.;

      real_type kinetic_energy = mass[i] * 
                                (vel_x[i] * vel_x[i] +
                                 vel_y[i] * vel_y[i] +
                                 vel_z[i] * vel_z[i]); // 7flops
      // reductionAtomics1
      sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_energy(energy_acc[0]);
      atomic_energy.fetch_add(kinetic_energy);
    });
  }).wait();

  // read energy value
  auto energy_host = energy_buf.get_access<sycl::access::mode::read>();
  return energy_host[0];
}
#endif

void GSimulation ::start() {
  real_type energy;
  real_type dt = get_tstep();
  int n = get_npart();

  init(n);

  print_header();

  _totTime = 0.;

  CPUTime time;
  double ts0 = 0;
  double ts1 = 0;
  double nd = double(n);
  double gflops = 1e-9 * ((11. + 18.) * nd * nd + nd * 19.);
  double av = 0.0, dev = 0.0;
  int nf = 0;

  const double t0 = time.start();
  for (int s = 1; s <= get_nsteps(); ++s) {
    ts0 += time.start();

#ifdef GPU
    get_acceleration_kernel(n);
#else
    get_acceleration(n);
#endif

#ifdef GPU
    energy = updateParticles_kernel(n, dt);
#else
    energy = updateParticles(n, dt);
#endif
    _kenergy = 0.5 * energy;

    ts1 += time.stop();
    if (!(s % get_sfreq())) {
      nf += 1;
      std::cout << " " << std::left << std::setw(8) << s << std::left
                << std::setprecision(5) << std::setw(8) << s * get_tstep()
                << std::left << std::setprecision(5) << std::setw(12)
                << _kenergy << std::left << std::setprecision(5)
                << std::setw(12) << (ts1 - ts0) << std::left
                << std::setprecision(5) << std::setw(12)
                << gflops * get_sfreq() / (ts1 - ts0) << std::endl;
      if (nf > 2) {
        av += gflops * get_sfreq() / (ts1 - ts0);
        dev += gflops * get_sfreq() * gflops * get_sfreq() /
               ((ts1 - ts0) * (ts1 - ts0));
      }

      ts0 = 0;
      ts1 = 0;
    }

  } // end of the time step loop

  const double t1 = time.stop();
  _totTime = (t1 - t0);
  _totFlops = gflops * get_nsteps();

  av /= (double)(nf - 2);
  dev = sqrt(dev / (double)(nf - 2) - av * av);

  std::cout << std::endl;
  std::cout << "# Total Time (s)      : " << _totTime << std::endl;
  std::cout << "# Average Performance : " << av << " +- " << dev << std::endl;
  std::cout << "===============================" << std::endl;
}

void GSimulation ::print_header() {
  std::cout << " nPart = " << get_npart() << "; "
            << "nSteps = " << get_nsteps() << "; "
            << "dt = " << get_tstep() << std::endl;

  std::cout << "------------------------------------------------" << std::endl;
  std::cout << " " << std::left << std::setw(8) << "s" << std::left
            << std::setw(8) << "dt" << std::left << std::setw(12) << "kenergy"
            << std::left << std::setw(12) << "time (s)" << std::left
            << std::setw(12) << "GFlops" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
}

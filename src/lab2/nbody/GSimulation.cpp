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
using namespace sycl;

GSimulation ::GSimulation() : particles(nullptr)
{
  std::cout << "===============================" << std::endl;
  std::cout << " Initialize Gravity Simulation" << std::endl;
  set_npart(16000);
  set_nsteps(10);
  set_tstep(0.1);
  set_sfreq(1);
}

void GSimulation ::set_number_of_particles(int N)
{
  set_npart(N);
}

void GSimulation ::set_number_of_steps(int N)
{
  set_nsteps(N);
}

void GSimulation ::init_pos()
{
  std::random_device rd; // random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(0, 1.0);

  for (int i = 0; i < get_npart(); ++i)
  {
#ifdef SOA
    {
      particles->pos_x[i] = unif_d(gen);
      particles->pos_y[i] = unif_d(gen);
      particles->pos_z[i] = unif_d(gen);
    }
#else
    {
      particles[i].pos[0] = unif_d(gen);
      particles[i].pos[1] = unif_d(gen);
      particles[i].pos[2] = unif_d(gen);
    }
#endif
  }
}

void GSimulation ::init_vel()
{
  std::random_device rd; // random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(-1.0, 1.0);

  for (int i = 0; i < get_npart(); ++i)
  {
#ifdef SOA
    {
      particles->vel_x[i] = unif_d(gen) * 1.0e-3f;
      particles->vel_y[i] = unif_d(gen) * 1.0e-3f;
      particles->vel_z[i] = unif_d(gen) * 1.0e-3f;
    }
#else
    {
      particles[i].vel[0] = unif_d(gen) * 1.0e-3f;
      particles[i].vel[1] = unif_d(gen) * 1.0e-3f;
      particles[i].vel[2] = unif_d(gen) * 1.0e-3f;
    }
#endif
  }
}

void GSimulation ::init_acc()
{
  for (int i = 0; i < get_npart(); ++i)
  {
#ifdef SOA
    {
      particles->acc_x[i] = 0.f;
      particles->acc_y[i] = 0.f;
      particles->acc_z[i] = 0.f;
    }
#else
    {
      particles[i].acc[0] = 0.f;
      particles[i].acc[1] = 0.f;
      particles[i].acc[2] = 0.f;
    }
#endif
  }
}

void GSimulation ::init_mass()
{
  real_type n = static_cast<real_type>(get_npart());
  std::random_device rd; // random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(0.0, 1.0);

  for (int i = 0; i < get_npart(); ++i)
  {
#ifdef SOA
    {
      particles->mass[i] = n * unif_d(gen);
    }
#else
    {
      particles[i].mass = n * unif_d(gen);
    }
#endif
  }
}

void GSimulation ::get_acceleration(int n)
{
  int i, j;

  const float softeningSquared = 1e-3f;
  const float G = 6.67259e-11f;

  for (i = 0; i < n; i++) // update acceleration
  {
    real_type ax_i = 0.f;
    real_type ay_i = 0.f;
    real_type az_i = 0.f;
#ifdef SOA
    {
      ax_i = particles->acc_x[i];
      ay_i = particles->acc_y[i];
      az_i = particles->acc_z[i];
    }
#else
    {
      ax_i = particles[i].acc[0];
      ay_i = particles[i].acc[1];
      az_i = particles[i].acc[2];
    }
#endif
    for (j = 0; j < n; j++)
    {
      real_type dx, dy, dz;
      real_type distanceSqr = 0.0f;
      real_type distanceInv = 0.0f;

#ifdef SOA
      {
        dx = particles->pos_x[j] - particles->pos_x[i]; // 1flop
        dy = particles->pos_y[j] - particles->pos_y[i]; // 1flop
        dz = particles->pos_z[j] - particles->pos_z[i]; // 1flop
      }
#else
      {
        dx = particles[j].pos[0] - particles[i].pos[0]; // 1flop
        dy = particles[j].pos[1] - particles[i].pos[1]; // 1flop
        dz = particles[j].pos[2] - particles[i].pos[2]; // 1flop
      }
#endif
      distanceSqr = dx * dx + dy * dy + dz * dz + softeningSquared; // 6flops
      distanceInv = 1.0f / sqrtf(distanceSqr);                      // 1div+1sqrt

#ifdef SOA
      {
        ax_i += dx * G * particles->mass[j] * distanceInv * distanceInv * distanceInv; // 6flops
        ay_i += dy * G * particles->mass[j] * distanceInv * distanceInv * distanceInv; // 6flops
        az_i += dz * G * particles->mass[j] * distanceInv * distanceInv * distanceInv; // 6flops
        particles->acc_x[i] = ax_i;
        particles->acc_y[i] = ay_i;
        particles->acc_z[i] = az_i;
      }
#else
      {
        ax_i += dx * G * particles[j].mass * distanceInv * distanceInv * distanceInv; // 6flops
        ay_i += dy * G * particles[j].mass * distanceInv * distanceInv * distanceInv; // 6flops
        az_i += dz * G * particles[j].mass * distanceInv * distanceInv * distanceInv; // 6flops
        particles[i].acc[0] = ax_i;
        particles[i].acc[1] = ay_i;
        particles[i].acc[2] = az_i;
      }
#endif
    }
  }
}

void GSimulation ::get_acceleration_kernel(sycl::queue Q, int n)
{
  const float softeningSquared = 1e-3f;
  const float G = 6.67259e-11f;

  Q.submit([&](handler &cgh)
             {
#ifdef SOA
        ParticleSoA* p = particles;
#else
        ParticleAoS* p = particles;
#endif
    cgh.parallel_for(n, [=](id<1> item) {
      int i = item[0];
      real_type ax_i = 0.f;
      real_type ay_i = 0.f;
      real_type az_i = 0.f;

      for (int j= 0; j< n; j++) 
      {
        real_type dx, dy, dz;
#ifdef SOA
                dx = p->pos_x[j] - p->pos_x[i];	//1flop
                dy = p->pos_y[j] - p->pos_y[i];	//1flop
                dz = p->pos_z[j] - p->pos_z[i];	//1flop
#else
                dx = p[j].pos[0] - p[i].pos[0];	//1flop
                dy = p[j].pos[1] - p[i].pos[1];	//1flop
                dz = p[j].pos[2] - p[i].pos[2];	//1flop
#endif
                real_type distanceSqr = dx * dx + dy * dy + dz * dz + softeningSquared;
                real_type distanceInv = 1.0f / sqrtf(distanceSqr);
#ifdef SOA
                real_type force = G * p->mass[j] * distanceInv * distanceInv * distanceInv;
#else
                real_type force = G * p[j].mass * distanceInv * distanceInv * distanceInv;
#endif
                ax_i += dx * force;
                ay_i += dy * force;
                az_i += dz * force;
      }
#ifdef SOA
            p->acc_x[i] = ax_i;
            p->acc_y[i] = ay_i;
            p->acc_z[i] = az_i;
#else
            p[i].acc[0] = ax_i;
            p[i].acc[1] = ay_i;
            p[i].acc[2] = az_i;
#endif
    }); })
      .wait();
}

real_type GSimulation ::updateParticles(int n, real_type dt)
{
  int i;
  real_type energy = 0;

  for (i = 0; i < n; ++i) // update position
  {
#ifdef SOA
    particles->vel_x[i] += particles->acc_x[i] * dt; // 2flops
    particles->vel_y[i] += particles->acc_y[i] * dt; // 2flops
    particles->vel_z[i] += particles->acc_z[i] * dt; // 2flops

    particles->pos_x[i] += particles->vel_x[i] * dt; // 2flops
    particles->pos_y[i] += particles->vel_y[i] * dt; // 2flops
    particles->pos_z[i] += particles->vel_z[i] * dt; // 2flops

    particles->acc_x[i] = 0.;
    particles->acc_y[i] = 0.;
    particles->acc_z[i] = 0.;
    energy += particles->mass[i] * (particles->vel_x[i] * particles->vel_x[i] +
                                    particles->vel_y[i] * particles->vel_y[i] +
                                    particles->vel_z[i] * particles->vel_z[i]); // 7flops
#else
    particles[i].vel[0] += particles[i].acc[0] * dt; // 2flops
    particles[i].vel[1] += particles[i].acc[1] * dt; // 2flops
    particles[i].vel[2] += particles[i].acc[2] * dt; // 2flops

    particles[i].pos[0] += particles[i].vel[0] * dt; // 2flops
    particles[i].pos[1] += particles[i].vel[1] * dt; // 2flops
    particles[i].pos[2] += particles[i].vel[2] * dt; // 2flops

    particles[i].acc[0] = 0.;
    particles[i].acc[1] = 0.;
    particles[i].acc[2] = 0.;
    energy += particles[i].mass * (particles[i].vel[0] * particles[i].vel[0] +
                                   particles[i].vel[1] * particles[i].vel[1] +
                                   particles[i].vel[2] * particles[i].vel[2]); // 7flops
#endif
  }
  return energy;
}

real_type GSimulation ::updateParticlesKernel(sycl::queue Q, int n, real_type dt)
{
  real_type energy = 0;
  sycl::buffer<real_type> energyBuffer(&energy, 1);

  Q.submit([&](handler &cgh)
             {
#ifdef SOA
    ParticleSoA* p = particles;
#else
    ParticleAoS *p = particles;
#endif

    auto e = energyBuffer.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for(n, [=](id<1> i) {
#ifdef SOA
    p->vel_x[i] += p->acc_x[i] * dt; //2flops
    p->vel_y[i] += p->acc_y[i] * dt; //2flops
    p->vel_z[i] += p->acc_z[i] * dt; //2flops

    p->pos_x[i] += p->vel_x[i] * dt; //2flops
    p->pos_y[i] += p->vel_y[i] * dt; //2flops
    p->pos_z[i] += p->vel_z[i] * dt; //2flops

    p->acc_x[i] = 0.;
    p->acc_y[i] = 0.;
    p->acc_z[i] = 0.;
    real_type v2 = p->vel_x[i]*p->vel_x[i] +
        p->vel_y[i]*p->vel_y[i] +
        p->vel_z[i]*p->vel_z[i];
   real_type tmpEnerg = (v2 * p->mass[i]);
#else
    p[i].vel[0] += p[i].acc[0] * dt; //2flops
    p[i].vel[1] += p[i].acc[1] * dt; //2flops
    p[i].vel[2] += p[i].acc[2] * dt; //2flops

    p[i].pos[0] += p[i].vel[0] * dt; //2flops
    p[i].pos[1] += p[i].vel[1] * dt; //2flops
    p[i].pos[2] += p[i].vel[2] * dt; //2flops

    p[i].acc[0] = 0.;
    p[i].acc[1] = 0.;
    p[i].acc[2] = 0.;

    real_type v2 = p[i].vel[0]*p[i].vel[0] +
            p[i].vel[1]*p[i].vel[1] +
            p[i].vel[2]*p[i].vel[2];
    real_type tmpEnerg = (v2 * p[i].mass);
#endif
    sycl::atomic_ref<real_type, sycl::memory_order::relaxed, sycl::memory_scope::device,
    sycl::access::address_space::global_space> atEn(e[0]);
    atEn.fetch_add(tmpEnerg); 
    }); })
      .wait();

  auto hostAcc = energyBuffer.get_host_access(sycl::read_only);
  energy = hostAcc[0];
  return energy;
}

void GSimulation ::start(bool gpu)
{
  real_type energy;
  real_type dt = get_tstep();
  int n = get_npart();
  std::cout << "1" << std::endl;

  _gpu = gpu;
  std::cout << "antes device" << std::endl;
  sycl::device _sD = sycl::device(sycl::cpu_selector_v);;
  std::cout << "antes q" << std::endl;
  std::cout << "1" << std::endl;
  if (_gpu)
  {
    _sD = sycl::device(sycl::gpu_selector_v);
  }
  std::cout << "1" << std::endl;
  // printf(">> Device: ", _sD.get_info<sycl::info::device::name>());
  sycl::queue _sQ = sycl::queue(_sD);

  // allocate particles
#ifdef SOA
  particles = sycl::malloc_shared<ParticleSoA>(n, _sQ);
  std::cout << "2" << std::endl;
  particles->init(n, _sQ); 
#else
  particles = sycl::malloc_shared<ParticleAoS>(n, _sQ);
  for (int i = 0; i < n; ++i)
    particles[i].init();
#endif
  std::cout << "2" << std::endl;
  init_pos();
  init_vel();
  init_acc();
  init_mass();

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
  for (int s = 1; s <= get_nsteps(); ++s)
  {
    ts0 += time.start();

    if (_gpu)
    {
      get_acceleration_kernel(_sQ, n);
      energy = updateParticlesKernel(_sQ, n, dt);
    }
    else
    {
      get_acceleration(n);
      energy = updateParticles(n, dt);
    }

    _kenergy = 0.5 * energy;

    ts1 += time.stop();
    if (!(s % get_sfreq()))
    {
      nf += 1;
      std::cout << " "
                << std::left << std::setw(8) << s
                << std::left << std::setprecision(5) << std::setw(8) << s * get_tstep()
                << std::left << std::setprecision(5) << std::setw(12) << _kenergy
                << std::left << std::setprecision(5) << std::setw(12) << (ts1 - ts0)
                << std::left << std::setprecision(5) << std::setw(12) << gflops * get_sfreq() / (ts1 - ts0)
                << std::endl;
      if (nf > 2)
      {
        av += gflops * get_sfreq() / (ts1 - ts0);
        dev += gflops * get_sfreq() * gflops * get_sfreq() / ((ts1 - ts0) * (ts1 - ts0));
      }

      ts0 = 0;
      ts1 = 0;
    }

  } // end of the time step loop

  const double t1 = time.stop();
  _totTime = (t1 - t0);
  _totFlops = gflops * get_nsteps();

  av /= (double)(nf - 2);
  dev = sycl::sqrt(dev / (double)(nf - 2) - av * av);

  std::cout << std::endl;
  std::cout << "# Total Time (s)      : " << _totTime << std::endl;
  std::cout << "# Average Performance : " << av << " +- " << dev << std::endl;
  std::cout << "===============================" << std::endl;
}

void GSimulation ::print_header()
{

  std::cout << " nPart = " << get_npart() << "; "
            << "nSteps = " << get_nsteps() << "; "
            << "dt = " << get_tstep() << std::endl;

  std::cout << "------------------------------------------------" << std::endl;
  std::cout << " "
            << std::left << std::setw(8) << "s"
            << std::left << std::setw(8) << "dt"
            << std::left << std::setw(12) << "kenergy"
            << std::left << std::setw(12) << "time (s)"
            << std::left << std::setw(12) << "GFlops"
            << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
}

GSimulation::~GSimulation()
{
  if (particles)
  {
#ifdef SOA
    particles->~ParticleSoA();
#endif
    // sycl::free(particles, _sQ);
    particles = nullptr;
  }
}

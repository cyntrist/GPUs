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
#include <CL/sycl.hpp>

using  namespace  cl::sycl;

GSimulation :: GSimulation()
{
  std::cout << "===============================" << std::endl;
  std::cout << " Initialize Gravity Simulation" << std::endl;
  set_npart(16000);
  set_nsteps(10);
  set_tstep(0.1);
  set_sfreq(1);
}

void GSimulation::setSyclQueue(sycl::queue Q) {
    _syclQueue = Q;
}

void GSimulation :: set_number_of_particles(int N)
{
  set_npart(N);
}

void GSimulation :: set_number_of_steps(int N)
{
  set_nsteps(N);
}

void GSimulation :: init_pos()
{
  std::random_device rd;	//random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(0,1.0);

  for(int i=0; i<get_npart(); ++i)
  {
#ifdef SoA
    particles->pos_x[i] = unif_d(gen);
    particles->pos_y[i] = unif_d(gen);
    particles->pos_z[i] = unif_d(gen);
#else
    particles[i].pos[0] = unif_d(gen);
    particles[i].pos[1] = unif_d(gen);
    particles[i].pos[2] = unif_d(gen);
#endif
  }
}

void GSimulation :: init_vel()
{
  std::random_device rd;        //random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(-1.0,1.0);

  for(int i=0; i<get_npart(); ++i)
  {
#ifdef SoA
    particles->vel_x[i] = unif_d(gen) * 1.0e-3f;
    particles->vel_y[i] = unif_d(gen) * 1.0e-3f;
    particles->vel_z[i] = unif_d(gen) * 1.0e-3f;
#else
    particles[i].vel[0] = unif_d(gen) * 1.0e-3f;
    particles[i].vel[1] = unif_d(gen) * 1.0e-3f;
    particles[i].vel[2] = unif_d(gen) * 1.0e-3f;
#endif
  }
}

void GSimulation :: init_acc()
{
  for(int i=0; i<get_npart(); ++i)
  {
#ifdef SoA
    particles->acc_x[i] = 0.0f;
    particles->acc_y[i] = 0.0f;
    particles->acc_z[i] = 0.0f;
#else
    particles[i].acc[0] = 0.f;
    particles[i].acc[1] = 0.f;
    particles[i].acc[2] = 0.f;
#endif
  }
}

void GSimulation :: init_mass()
{
  real_type n   = static_cast<real_type> (get_npart());
  std::random_device rd;        //random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(0.0,1.0);

  for(int i=0; i<get_npart(); ++i)
  {
#ifdef SoA
    particles->mass[i] = n * unif_d(gen);
#else
    particles[i].mass = n * unif_d(gen);
#endif
  }
}

void GSimulation :: get_acceleration(int n)
{
   int i,j;

   const float softeningSquared = 1e-3f;
   const float G = 6.67259e-11f;

   for (i = 0; i < n; i++)// update acceleration
   {
#ifdef SoA
    real_type ax_i = particles->acc_x[i];
    real_type ay_i = particles->acc_y[i];
    real_type az_i = particles->acc_z[i];
#else
    real_type ax_i = particles[i].acc[0];
    real_type ay_i = particles[i].acc[1];
    real_type az_i = particles[i].acc[2];
#endif
    for (j = 0; j < n; j++)
    {
        real_type dx, dy, dz;
        real_type distanceSqr = 0.0f;
        real_type distanceInv = 0.0f;

#ifdef SoA
        dx = particles->pos_x[j] - particles->pos_x[i];	//1flop
        dy = particles->pos_y[j] - particles->pos_y[i];	//1flop
        dz = particles->pos_z[j] - particles->pos_z[i];	//1flop
#else
        dx = particles[j].pos[0] - particles[i].pos[0];	//1flop
        dy = particles[j].pos[1] - particles[i].pos[1];	//1flop
        dz = particles[j].pos[2] - particles[i].pos[2];	//1flop
#endif
        distanceSqr = dx*dx + dy*dy + dz*dz + softeningSquared;	//6flops
        distanceInv = 1.0f / sqrtf(distanceSqr);			//1div+1sqrt

#ifdef SoA
        ax_i += dx * G * particles->mass[j] * distanceInv * distanceInv * distanceInv; //6flops
        ay_i += dy * G * particles->mass[j] * distanceInv * distanceInv * distanceInv; //6flops
        az_i += dz * G * particles->mass[j] * distanceInv * distanceInv * distanceInv; //6flops
#else
        ax_i += dx * G * particles[j].mass * distanceInv * distanceInv * distanceInv; //6flops
        ay_i += dy * G * particles[j].mass * distanceInv * distanceInv * distanceInv; //6flops
        az_i += dz * G * particles[j].mass * distanceInv * distanceInv * distanceInv; //6flops
#endif
    }
#ifdef SoA
    particles->acc_x[i] = ax_i;
    particles->acc_y[i] = ay_i;
    particles->acc_z[i] = az_i;
#else
    particles[i].acc[0] = ax_i;
    particles[i].acc[1] = ay_i;
    particles[i].acc[2] = az_i;
#endif
   }
}

void GSimulation::get_acceleration_SYCL(int n) {
    float softeningSquared = 1e-3f;
    float G = 6.67259e-11f;

    _syclQueue.submit([&](handler &h) {
#ifdef SoA
        ParticleSoA* local_particles = particles;
#else
        ParticleAoS* local_particles = particles;
#endif
        h.parallel_for(n, [=](id<1> item) {
            int i = item[0];
            real_type ax_i = 0.0f, ay_i = 0.0f, az_i = 0.0f;

            for (int j = 0; j < n; j++) {

                real_type dx, dy, dz;
#ifdef SoA
                dx = local_particles->pos_x[j] - local_particles->pos_x[i];	//1flop
                dy = local_particles->pos_y[j] - local_particles->pos_y[i];	//1flop
                dz = local_particles->pos_z[j] - local_particles->pos_z[i];	//1flop
#else
                dx = local_particles[j].pos[0] - local_particles[i].pos[0];	//1flop
                dy = local_particles[j].pos[1] - local_particles[i].pos[1];	//1flop
                dz = local_particles[j].pos[2] - local_particles[i].pos[2];	//1flop
#endif

                real_type distanceSqr = dx * dx + dy * dy + dz * dz + softeningSquared;
                real_type distanceInv = 1.0f / sqrtf(distanceSqr);

#ifdef SoA
                real_type force = G * local_particles->mass[j] * distanceInv * distanceInv * distanceInv;
#else
                real_type force = G * local_particles[j].mass * distanceInv * distanceInv * distanceInv;
#endif
                ax_i += dx * force;
                ay_i += dy * force;
                az_i += dz * force;
            }

#ifdef SoA
            local_particles->acc_x[i] = ax_i;
            local_particles->acc_y[i] = ay_i;
            local_particles->acc_z[i] = az_i;
#else
            local_particles[i].acc[0] = ax_i;
            local_particles[i].acc[1] = ay_i;
            local_particles[i].acc[2] = az_i;
#endif
        });
    }).wait();
}


real_type GSimulation :: updateParticles(int n, real_type dt)
{
   int i;
   real_type energy = 0;

   for (i = 0; i < n; ++i)// update position
   {
#ifdef SoA
    particles->vel_x[i] += particles->acc_x[i] * dt; //2flops
    particles->vel_y[i] += particles->acc_y[i] * dt; //2flops
    particles->vel_z[i] += particles->acc_z[i] * dt; //2flops

    particles->pos_x[i] += particles->vel_x[i] * dt; //2flops
    particles->pos_y[i] += particles->vel_y[i] * dt; //2flops
    particles->pos_z[i] += particles->vel_z[i] * dt; //2flops

    particles->acc_x[i] = 0.;
    particles->acc_y[i] = 0.;
    particles->acc_z[i] = 0.;

    energy += particles->mass[i] * (
    particles->vel_x[i]*particles->vel_x[i] +
            particles->vel_y[i]*particles->vel_y[i] +
            particles->vel_z[i]*particles->vel_z[i]);
#else
    particles[i].vel[0] += particles[i].acc[0] * dt; //2flops
    particles[i].vel[1] += particles[i].acc[1] * dt; //2flops
    particles[i].vel[2] += particles[i].acc[2] * dt; //2flops

    particles[i].pos[0] += particles[i].vel[0] * dt; //2flops
    particles[i].pos[1] += particles[i].vel[1] * dt; //2flops
    particles[i].pos[2] += particles[i].vel[2] * dt; //2flops

    particles[i].acc[0] = 0.;
    particles[i].acc[1] = 0.;
    particles[i].acc[2] = 0.;

    energy += particles[i].mass * (
      particles[i].vel[0]*particles[i].vel[0] +
            particles[i].vel[1]*particles[i].vel[1] +
            particles[i].vel[2]*particles[i].vel[2]); //7flops
#endif
   }
   return energy;
}

real_type GSimulation::updateParticles_SYCL(int n, real_type dt) {
    real_type ogEnergy = 0.0f;
    sycl::buffer<real_type> energyBuffer(&ogEnergy, 1);

    _syclQueue.submit([&](handler &h) {
        auto energy = energyBuffer.get_access<sycl::access::mode::write>(h);
#ifdef SoA
        ParticleSoA* local_particles = particles;
#else
        ParticleAoS* local_particles = particles;
#endif
        h.parallel_for(n, [=](id<1> i) {
#ifdef SoA
            local_particles->vel_x[i] += local_particles->acc_x[i] * dt; //2flops
            local_particles->vel_y[i] += local_particles->acc_y[i] * dt; //2flops
            local_particles->vel_z[i] += local_particles->acc_z[i] * dt; //2flops

            local_particles->pos_x[i] += local_particles->vel_x[i] * dt; //2flops
            local_particles->pos_y[i] += local_particles->vel_y[i] * dt; //2flops
            local_particles->pos_z[i] += local_particles->vel_z[i] * dt; //2flops

            local_particles->acc_x[i] = 0.;
            local_particles->acc_y[i] = 0.;
            local_particles->acc_z[i] = 0.;

            real_type v2 = local_particles->vel_x[i]*local_particles->vel_x[i] +
                    local_particles->vel_y[i]*local_particles->vel_y[i] +
                    local_particles->vel_z[i]*local_particles->vel_z[i];
            real_type tmpEnerg = (v2 * local_particles->mass[i]);
#else
            local_particles[i].vel[0] += local_particles[i].acc[0] * dt; //2flops
            local_particles[i].vel[1] += local_particles[i].acc[1] * dt; //2flops
            local_particles[i].vel[2] += local_particles[i].acc[2] * dt; //2flops

            local_particles[i].pos[0] += local_particles[i].vel[0] * dt; //2flops
            local_particles[i].pos[1] += local_particles[i].vel[1] * dt; //2flops
            local_particles[i].pos[2] += local_particles[i].vel[2] * dt; //2flops

            local_particles[i].acc[0] = 0.;
            local_particles[i].acc[1] = 0.;
            local_particles[i].acc[2] = 0.;

            real_type v2 = local_particles[i].vel[0]*local_particles[i].vel[0] +
                    local_particles[i].vel[1]*local_particles[i].vel[1] +
                    local_particles[i].vel[2]*local_particles[i].vel[2];
            real_type tmpEnerg = (v2 * local_particles[i].mass);
#endif

            sycl::atomic_ref<real_type, sycl::memory_order::relaxed, sycl::memory_scope::device,
            sycl::access::address_space::global_space> atEn(energy[0]);
            atEn.fetch_add(tmpEnerg);
        });
    }).wait();

    auto energy = energyBuffer.get_access<sycl::access::mode::read>();
    return energy[0];
}

void GSimulation :: start(bool useSycl)
{
  real_type energy;
  real_type dt = get_tstep();
  int n = get_npart();

  //allocate particles
  #ifdef SoA
  particles = malloc_shared<ParticleSoA>(1, _syclQueue);
  particles->init(n, _syclQueue);
  #else
  particles = malloc_shared<ParticleAoS>(n, _syclQueue);
  for(int i = 0; i < n; ++i) {
      particles[i].init();
  }
  #endif

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
  double gflops = 1e-9 * ( (11. + 18. ) * nd*nd  +  nd * 19. );
  double av=0.0, dev=0.0;
  int nf = 0;

  const double t0 = time.start();
  for (int s=1; s<=get_nsteps(); ++s)
  {
   ts0 += time.start();

    useSycl ? get_acceleration_SYCL(n) : get_acceleration(n);

    energy = useSycl ? updateParticles_SYCL(n, dt) : updateParticles(n, dt);
    _kenergy = 0.5 * energy;

    ts1 += time.stop();
    if(!(s%get_sfreq()) )
    {
      nf += 1;
      std::cout << " "
		<<  std::left << std::setw(8)  << s
		<<  std::left << std::setprecision(5) << std::setw(8)  << s*get_tstep()
		<<  std::left << std::setprecision(5) << std::setw(12) << _kenergy
		<<  std::left << std::setprecision(5) << std::setw(12) << (ts1 - ts0)
		<<  std::left << std::setprecision(5) << std::setw(12) << gflops*get_sfreq()/(ts1 - ts0)
		<<  std::endl;
      if(nf > 2)
      {
	av  += gflops*get_sfreq()/(ts1 - ts0);
	dev += gflops*get_sfreq()*gflops*get_sfreq()/((ts1-ts0)*(ts1-ts0));
      }

      ts0 = 0;
      ts1 = 0;
    }

  } //end of the time step loop

  const double t1 = time.stop();
  _totTime  = (t1-t0);
  _totFlops = gflops*get_nsteps();

  av/=(double)(nf-2);
  dev=std::sqrt((double)(dev/(double)(nf-2)-av*av));


  std::cout << std::endl;
  std::cout << "# Total Time (s)      : " << _totTime << std::endl;
  std::cout << "# Average Performance : " << av << " +- " <<  dev << std::endl;
  std::cout << "===============================" << std::endl;

}


void GSimulation :: print_header()
{

  std::cout << " nPart = " << get_npart()  << "; "
	    << "nSteps = " << get_nsteps() << "; "
	    << "dt = "     << get_tstep()  << std::endl;

  std::cout << "------------------------------------------------" << std::endl;
  std::cout << " "
	    <<  std::left << std::setw(8)  << "s"
	    <<  std::left << std::setw(8)  << "dt"
	    <<  std::left << std::setw(12) << "kenergy"
	    <<  std::left << std::setw(12) << "time (s)"
	    <<  std::left << std::setw(12) << "GFlops"
	    <<  std::endl;
  std::cout << "------------------------------------------------" << std::endl;


}

GSimulation :: ~GSimulation()
{
#ifdef SoA
    particles->~ParticleSoA();
#endif
  free(particles, _syclQueue);
  particles = nullptr;
}

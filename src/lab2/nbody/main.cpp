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

#include <iostream>
#include <exception>

#include "GSimulation.hpp"

int main(int argc, char** argv) 
{
  try
  {
    int N = 0;			//number of particles
    int nstep = 0; 		//number ot integration steps
    
    GSimulation sim;
    bool gpu = false;  

    if(argc>1)
    {
      N=atoi(argv[1]);
      sim.set_number_of_particles(N);  
      if(argc== 3 || argc == 4) 
      {
        nstep=atoi(argv[2]);
        sim.set_number_of_steps(nstep);  
      }
      if(argc == 4 && argv[3][0] == 'g') 
      {
        gpu = true; 
      }
    }

    sim.start(gpu);
    return 0;
  }
  catch (const sycl::exception& e)
  {
    std::cerr << "SYCL error: " << e.what() << std::endl;
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  return 1;
}

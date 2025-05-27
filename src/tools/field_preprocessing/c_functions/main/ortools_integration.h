//
// Created by corde on 4/20/2025.
//

#ifndef ORTOOLS_INTEGRATION_H
#define ORTOOLS_INTEGRATION_H

#endif //ORTOOLS_INTEGRATION_H

#include "ortools/constraint_solver/routing.h"
#include "ortools/constraint_solver/routing_parameters.h"
#include <vector>
#include <iostream>

std::vector<int> return_solution(const operations_research::RoutingIndexManager& manager,
                                 const operations_research::RoutingModel& routing,
                                 const operations_research::Assignment& solution,
                                 int num_vehicles);


std::vector<int> SolveRouting(const std::vector<std::vector<double>>& distance_matrix,
                              int num_vehicles,
                              const std::vector<int>& starts,
                              const std::vector<int>& ends);



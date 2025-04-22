#include "ortools/constraint_solver/routing.h"
#include "ortools/constraint_solver/routing_parameters.h"
#include <vector>
#include <iostream>




// The distance callback wrapper
class DistanceCallback {
public:
    DistanceCallback(const std::vector<std::vector<double>>& matrix,
                     const operations_research::RoutingIndexManager& manager)
        : matrix_(matrix), manager_(manager) {}

    int64_t operator()(int64_t from_index, int64_t to_index) const {
        auto from_node = manager_.IndexToNode(from_index).value();
        auto to_node = manager_.IndexToNode(to_index).value();
        return static_cast<int64_t>(matrix_[from_node][to_node]);  // Convert to int64_t here
    }

private:
    const std::vector<std::vector<double>>& matrix_;
    const operations_research::RoutingIndexManager& manager_;
};


std::vector<int> return_solution(const operations_research::RoutingIndexManager& manager,
                                 const operations_research::RoutingModel& routing,
                                 const operations_research::Assignment& solution,
                                 int num_vehicles) {
    std::vector<int> path_seq;
    std::cout << "Objective: " << solution.ObjectiveValue() << "\n";

    for (int vehicle_id = 0; vehicle_id < num_vehicles; ++vehicle_id) {
        if (!routing.IsVehicleUsed(solution, vehicle_id)) {
            continue;
        }
        int64_t index = routing.Start(vehicle_id);
        std::ostringstream plan_output;
        plan_output << "Route for vehicle " << vehicle_id << ":\n";
        int64_t route_distance = 0;

        while (!routing.IsEnd(index)) {
            plan_output << manager.IndexToNode(index) << " -> ";
            path_seq.push_back(manager.IndexToNode(index).value());
            int64_t previous_index = index;
            index = solution.Value(routing.NextVar(index));
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id);
        }
        plan_output << manager.IndexToNode(index) << "\n";
        path_seq.push_back(manager.IndexToNode(index).value());
        // plan_output << "Distance of the route: " << route_distance << "m\n";
        // std::cout << plan_output.str();
        // max_route_distance = std::max(route_distance, max_route_distance);
    }
    // std::cout << "Maximum of the route distances: " << max_route_distance << "m\n";
    return path_seq;
}

std::vector<int> SolveRouting(const std::vector<std::vector<double>>& distance_matrix,
                              int num_vehicles,
                              const std::vector<int>& starts,
                              const std::vector<int>& ends) {

    using namespace operations_research;

    std::vector<RoutingIndexManager::NodeIndex> start_indices;
    std::vector<RoutingIndexManager::NodeIndex> end_indices;

    for (int idx : starts)
        start_indices.emplace_back(RoutingIndexManager::NodeIndex(idx));

    for (int idx : ends)
        end_indices.emplace_back(RoutingIndexManager::NodeIndex(idx));

    RoutingIndexManager manager(distance_matrix.size(), num_vehicles, start_indices, end_indices);
    RoutingModel routing(manager);

    DistanceCallback callback(distance_matrix, manager);
    const int transit_callback_index = routing.RegisterTransitCallback(
        [&callback](int64_t from_index, int64_t to_index) -> int64_t {
            return callback(from_index, to_index);
        });

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index);

    RoutingSearchParameters parameters = DefaultRoutingSearchParameters();
    parameters.set_first_solution_strategy(FirstSolutionStrategy::PATH_CHEAPEST_ARC);
    parameters.set_local_search_metaheuristic(LocalSearchMetaheuristic::GUIDED_LOCAL_SEARCH);
    parameters.mutable_time_limit()->set_seconds(60);
    parameters.set_log_search(true);

    const Assignment* solution = nullptr;


    solution = routing.SolveWithParameters(parameters);

    if (solution) {
        std::cout << "Solution found! Returning solution list...\n";
        std::vector<int> path_seq = return_solution(manager, routing, *solution, num_vehicles);
        return path_seq;
    } else {
        std::cout << "No solution found.\n";
    }
}

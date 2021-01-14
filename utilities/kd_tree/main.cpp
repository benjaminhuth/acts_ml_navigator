#include <vector>
#include <iostream>
#include <random>

#include "kd_tree.hpp"


// Create Tree
constexpr int D = 10;

int main()
{
    std::srand(std::random_device{}());
    
    std::vector< kd_tree::Node<D>::Point > points;
    for(auto i=0ul; i<20000; ++i)
        points.push_back( kd_tree::Node<D>::Point::Random() );
    
    const auto tree = kd_tree::Node<D>::build_tree(points);
    
    // Test
    const int n_test = 1000;
    
    std::vector< kd_tree::Node<D>::Point > test_targets;
    
    for(int i=0; i<n_test; ++i)
    {
        // const auto target = kd_tree::Node<D>::Point(1,4);
        const kd_tree::Node<D>::Point target = kd_tree::Node<D>::Point::Random();
        
        // Find neighbor by loop
        std::vector< kd_tree::Node<D>::Scalar > distances;
        std::ranges::transform(points, std::back_inserter(distances), [&](auto &a){ return (target - a).dot(target - a); });
        const auto min_idx = std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));        
        const auto nn_loop = points[static_cast<std::size_t>(min_idx)];
        
        // Use kd-tree
        const auto nn_tree = std::get<0>(tree->query_neighbor(target));
        
        if( !nn_loop.isApprox(nn_tree) )
        {
            std::cout << "loop:   " << nn_loop.transpose()  << " - dist: " << (nn_loop - target).norm() << "\n";
            std::cout << "tree:   " << nn_tree.transpose() << " - dist: " << (nn_tree - target).norm() << std::endl;
            throw std::runtime_error("result not correct");
        }
        else
            std::cout << "passed test #" << i << std::endl;
    }   
}

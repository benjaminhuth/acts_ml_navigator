#ifndef KD_TREE_HPP
#define KD_TREE_HPP

#include <memory>
#include <span>
#include <ranges>
#include <algorithm>

#include <Eigen/Core>

namespace kd_tree
{
    template<int D, typename scalar_t = float>
    class Node
    {
    public:
        using Scalar = scalar_t;
        using NodePtr = std::unique_ptr<Node>;
        using Point = Eigen::Matrix<Scalar, D, 1>;
        using Index = std::size_t;
        using PointIndexTuple = std::tuple<Point, Index>;
        
    private:
        const PointIndexTuple m_val;
        const NodePtr m_left;
        const NodePtr m_right;
        Node *parent;
        
    public:
        Node() = delete;
        Node(const Node &) = delete;
        Node &operator=(const Node &) = delete;
        
        /// Constructor for leaf node
        Node(const PointIndexTuple & v) : 
            m_val(v) 
        {
        }
        
        /// Constructor for node with only one child
        Node(const PointIndexTuple & v, NodePtr && l) : 
            m_val(v), 
            m_left(std::move(l)) 
        {
            m_left->parent = this;
        }
        
        /// Constructor for node with two childs
        Node(const PointIndexTuple & v, NodePtr && l, NodePtr && r) : 
            m_val(v), 
            m_left(std::move(l)), 
            m_right(std::move(r)) 
        {
            m_left->parent = this;
            m_right->parent = this;
        }
        
        auto &point() const
        {
            return std::get<Point>(m_val);
        }
        
        auto index() const
        {
            return std::get<Index>(m_val);
        }
        
        static auto build_tree(const std::vector<Point> &points)
        {
            std::vector<Index> idxs(points.size());
            std::iota(idxs.begin(), idxs.end(), static_cast<Index>(0));
            
            std::vector<PointIndexTuple> transformed(points.size());
            
            std::ranges::transform(points, idxs, transformed.begin(), [](auto p, auto i){ return std::make_tuple(p,i); });
            
            return build_tree_impl(transformed, 0);
        }
        
        auto query_neighbor(const Point &target)
        {
            return std::get<const Node *>(query_neighbor_impl(target, 0))->m_val;
        }
        
    private:
        /// Builds a kd-tree out of a set of points
        static auto build_tree_impl(std::span<PointIndexTuple> points, const int d)
        {
            std::ranges::sort(points, [&](const auto &a, const auto &b)
            { 
                return std::get<Point>(a)[d] < std::get<Point>(b)[d];
            });
            
            if( points.size() == 1ul )
            {
                return std::make_unique<Node>(points[0]);
            }
            else if( points.size() == 2ul )
            {
                auto lnode = std::make_unique<Node>(points[1]);
                
                return std::make_unique<Node>(points[0], std::move(lnode));
            }
            else
            {
                const auto sep = static_cast<std::ptrdiff_t>((points.size() - 1ul) / 2ul);
                
                auto lnode = build_tree_impl(std::span<PointIndexTuple>(points.begin(), points.begin() + sep), (d+1) % D);
                auto rnode = build_tree_impl(std::span<PointIndexTuple>(points.begin() + sep + 1, points.end()), (d+1) % D);
                
                return std::make_unique<Node>(points[static_cast<std::size_t>(sep)], std::move(lnode), std::move(rnode));
            }
        }
        
        /// Find the next neighbor of a target
        std::tuple<const Node *, Scalar> query_neighbor_impl(const Point& target, const int d = 0) const
        {
            auto query_node = [&target, d](auto base_node, auto dist_to_base, auto test_node)
            {
                auto [neighbor_a, dist_to_a] = test_node->query_neighbor_impl(target, (d + 1) % D);
                
                auto best = dist_to_base < dist_to_a ? base_node : neighbor_a;
                auto dist_to_best = std::min(dist_to_base, dist_to_a);
            
                return std::make_tuple(best, dist_to_best);
            };
                
            const auto this_point = std::get<Point>(m_val);
            const auto dist_to_this = (target - this_point).dot(target - this_point);
            const bool next_is_left = target[d] < this_point[d];
            
            if( m_left && m_right )
            {
                const auto node_a = next_is_left ? m_left.get() : m_right.get();
                const auto node_b = next_is_left ? m_right.get() : m_left.get();
                
                const auto [best, dist_to_best] = query_node(this, dist_to_this, node_a);
                
                if( (target[d] - this_point[d])*(target[d] - this_point[d]) < dist_to_best )
                {
                    return query_node(best, dist_to_best, node_b);
                }
                
                return std::make_tuple(best, dist_to_best);
            }
            else if( m_left || m_right )
            {
                const auto node = m_left ? m_left.get() : m_right.get();
                
                return query_node(this, dist_to_this, node);                
            }
            else
            {
                return std::make_tuple(this, dist_to_this);
            }
        }
    };
}

#endif

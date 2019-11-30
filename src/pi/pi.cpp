#include <iostream>

#include "boost/compute/function.hpp"
#include "boost/compute/system.hpp"
#include "boost/compute/algorithm/count_if.hpp"
#include "boost/compute/container/vector.hpp"
#include "boost/compute/iterator/buffer_iterator.hpp"
#include "boost/compute/random/default_random_engine.hpp"
#include "boost/compute/random/uniform_real_distribution.hpp"
#include "boost/compute/types/fundamental.hpp"

using std::cout; using std::endl;
namespace compute = boost::compute;

int main()
{
    compute::device device = compute::system::default_device();

    cout << "device: " << device.name() << endl;
    cout << "platform: " << device.platform().name() << endl;

    compute::context context(device);
    compute::command_queue queue(context, device);

    using compute::float_;
    using compute::float2_;

    // one million random points
    size_t n = 1000000;

    // generate random numbers
    compute::default_random_engine random(queue);
    random.seed((unsigned int)std::time(0), queue);
    compute::uniform_real_distribution<float_> dist(0.0, 1.0);
    compute::vector<float_> vector(n * 2, context);
    dist.generate(vector.begin(), vector.end(), random, queue);

    // function returing true if the point is within the unit circle
    BOOST_COMPUTE_FUNCTION(bool, is_in_unit_circle, (const float2_ point),
    {
        const float x = point.x;
        const float y = point.y;

        return (x*x + y*y) < 1.0f;
    });

    // iterate over vector<float> as vector<float2>
    compute::buffer_iterator<float2_> start =
        compute::make_buffer_iterator<float2_>(vector.get_buffer(), 0);
    compute::buffer_iterator<float2_> end =
        compute::make_buffer_iterator<float2_>(vector.get_buffer(), vector.size() / 2);

    // count number of random points within the unit circle
    size_t count = compute::count_if(start, end, is_in_unit_circle, queue);

    // print out values
    float count_f = static_cast<float>(count);
    std::cout << "count: " << count << " / " << n << std::endl;
    std::cout << "ratio: " << count_f / float(n) << std::endl;
    std::cout << "pi = " << (count_f / float(n)) * 4.0f << std::endl;

    return 0;
}

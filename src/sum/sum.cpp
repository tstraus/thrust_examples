#include <iostream>

#include "boost/compute/core.hpp"
#include "boost/compute/algorithm.hpp"
#include "boost/compute/container/vector.hpp"
#include "boost/compute/random/default_random_engine.hpp"
#include "boost/compute/random/uniform_int_distribution.hpp"

using std::cout; using std::endl;
namespace compute = boost::compute;

int main(void)
{
    compute::device device = compute::system::default_device();

    cout << "device: " << device.name() << endl;
    cout << "platform: " << device.platform().name() << endl;

    compute::context context(device);
    compute::command_queue queue(context, device);

    compute::vector<int> dv(10000, context);
    compute::default_random_engine random(queue);
    random.seed((unsigned int)std::time(0), queue);
    compute::uniform_int_distribution<int> dist(0, 9999);

    dist.generate(dv.begin(), dv.end(), random, queue);

    int sum = 0;
    compute::reduce(dv.begin(), dv.end(), &sum, compute::plus<int>(), queue);

    cout << "sum: " << sum << endl;

    return 0;
}


//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/generate.h>
//#include <thrust/reduce.h>
//#include <thrust/functional.h>
//#include <thrust/random.h>
//
//int my_rand (void)
//{
//    static thrust::default_random_engine rng;
//    static thrust::uniform_int_distribution<int> dist(0, 9999);
//
//    return dist(rng);
//}
//
//int main(void)
//{
//    for (int i = 0; i < 100; i++)
//    {
//        // generate random data on the host
//        thrust::host_vector<int> h_vec(1000000);
//        thrust::generate(h_vec.begin(), h_vec.end(), my_rand);
//
//        // transfer to device and compute sum
//        thrust::device_vector<int> d_vec = h_vec;
//
//        // initial value of the reduction
//        int init = 0; 
//
//        // binary operation used to reduce values
//        thrust::plus<int> binary_op;
//
//        // compute sum on the device
//        int sum = thrust::reduce(d_vec.begin(), d_vec.end(), init, binary_op);
//
//        // print the sum
//        std::cout << "sum is " << sum << std::endl;
//    }
//
//    return 0;
//}

#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <iostream>
#include <iomanip>
#include <cmath>

// we could vary M & N to find the perf sweet spot

__host__ __device__
unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);

    return a;
}

struct estimate_pi : public thrust::unary_function<unsigned int,float>
{
    __host__ __device__
    float operator()(unsigned int thread_id)
    {
        float sum = 0;
        unsigned int N = 1000000; // samples per thread

        unsigned int seed = hash(thread_id);

        // seed a random number generator
        thrust::default_random_engine rng(seed);

        // create a mapping from random numbers to [0,1)
        thrust::uniform_real_distribution<float> u01(0,1);

        // take N samples in a quarter circle
        for(unsigned int i = 0; i < N; ++i)
        {
            // draw a sample from the unit square
            float x = u01(rng);
            float y = u01(rng);

            // measure distance from the origin
            float dist = sqrtf(x*x + y*y);

            // add 1.0f if (u0,u1) is inside the quarter circle
            if(dist <= 1.0f)
            sum += 1.0f;
        }

        // multiply by 4 to get the area of the whole circle
        sum *= 4.0f;

        // divide by N
        return sum / N;
    }
};

int main(void)
{
    // use 30K independent seeds
    int M = 30000;

    float estimate = thrust::transform_reduce(thrust::counting_iterator<int>(0),
    thrust::counting_iterator<int>(M),
    estimate_pi(),
    0.0f,
    thrust::plus<float>());

    estimate /= M;

    std::cout << std::setprecision(16);
    std::cout << "pi is approximately " << estimate << std::endl;

    return 0;
}


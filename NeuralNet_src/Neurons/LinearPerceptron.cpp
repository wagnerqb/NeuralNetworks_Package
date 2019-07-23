/***************************************************************************//**
 * @file     LinearPerceptron.cpp
 * @date     05 Apr 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Neurons
 * @brief    Class that models the perceptron neuron with the Linear
 *           activation function.
 ******************************************************************************/

#include "LinearPerceptron.h"

// ActvF
double LinearPerceptron::ActvF(double u) {
    return u;
};

// dActvF_du
double LinearPerceptron::dActvF_du(double u){
    return 1.;
};

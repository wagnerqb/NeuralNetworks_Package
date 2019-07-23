/***************************************************************************//**
 * @file     HyperbolicTangentPerceptron.cpp
 * @date     05 Apr 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Neurons
 * @brief    Class that models the perceptron neuron with the HyperbolicTangent
 *           activation function.
 ******************************************************************************/

#include "HyperbolicTangentPerceptron.h"

// ActvF
double HyperbolicTangentPerceptron::ActvF(double u) {

    return ((1.-exp(-_beta*u))/(1. + exp(-_beta*u)));
};

// dActvF_du
double HyperbolicTangentPerceptron::dActvF_du(double u){

    return ((2.*_beta*exp(-_beta*u))/pow((1. + exp(-_beta*u)),2));
};

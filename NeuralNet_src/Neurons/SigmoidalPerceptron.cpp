/***************************************************************************//**
 * @file     SigmoidalPerceptron.cpp
 * @date     05 Apr 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Neurons
 * @brief    Class that models the perceptron neuron with the Sigmoidal
 *           activation function.
 ******************************************************************************/

#include "SigmoidalPerceptron.h"

// ActvF
double SigmoidalPerceptron::ActvF(double u) {

    return (1./(1. + exp(-_beta*u)));
};

// dActvF_du
double SigmoidalPerceptron::dActvF_du(double u){

    return ((_beta*exp(-_beta*u))/pow((1. + exp(-_beta*u)),2));
};

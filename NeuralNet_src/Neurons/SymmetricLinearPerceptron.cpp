/***************************************************************************//**
 * @file     SymmetricLinearPerceptron.cpp
 * @date     05 Apr 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Neurons
 * @brief    Class that models the perceptron neuron with the SymmetricLinear
 *           activation function.
 ******************************************************************************/

#include "SymmetricLinearPerceptron.h"

// ActvF
double SymmetricLinearPerceptron::ActvF(double u) {

    if (u > _a_slope) {
        return _a_slope;
    }
    else if (u < -_a_slope) {
        return - _a_slope;
    }
    else {
        return u;
    }

};

// dActvF_du
double SymmetricLinearPerceptron::dActvF_du(double u){

    if (u > _a_slope) {
        return 0.;
    }
    else if (u < -_a_slope) {
        return 0.;
    }
    else {
        return 1.;
    }

};

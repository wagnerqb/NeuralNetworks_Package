/***************************************************************************//**
 * @file     SHLPerceptron.cpp
 * @date     05 Apr 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Neurons
 * @brief    Class that models the perceptron neuron with the Symmetric Hard Limiter
 *           activation function.
 ******************************************************************************/

#include "SHLPerceptron.h"

// ActvF
double SHLPerceptron::ActvF(double u) {
    if (u > 0.) {
        return 1.;
    }
    else if (u == 0.) {
        return 0.;
    } else {
        return -1.;
    }

};

// dActvF_du
double SHLPerceptron::dActvF_du(double u){
    return 0.;
};

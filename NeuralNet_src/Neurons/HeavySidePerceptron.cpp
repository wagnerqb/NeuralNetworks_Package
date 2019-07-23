/***************************************************************************//**
 * @file     HeavySidePerceptron.cpp
 * @date     05 Apr 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Neurons
 * @brief    Class that models the perceptron neuron with the Heavyside
 *           activation function.
 ******************************************************************************/

#include "HeavySidePerceptron.h"

// ActvF
double HeavySidePerceptron::ActvF(double u) {
    if (u >= 0.) {
        return 1.;
    }
    else {
        return 0.;
    }

};

// dActvF_du
double HeavySidePerceptron::dActvF_du(double u){
    return 0.;
};

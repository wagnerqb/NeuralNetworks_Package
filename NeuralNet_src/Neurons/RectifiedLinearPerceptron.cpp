/***************************************************************************//**
 * @file     RectifiedLinearPerceptron.cpp
 * @date     05 Apr 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Neurons
 * @brief    Class that models the perceptron neuron with the Heavyside
 *           activation function.
 ******************************************************************************/

#include "RectifiedLinearPerceptron.h"

// ActvF
double RectifiedLinearPerceptron::ActvF(double u) {
    return fmax(0., u);

};

// dActvF_du
double RectifiedLinearPerceptron::dActvF_du(double u){
    if (u <= 0.) {
        return 0.;
    }
    else {
        return 1.;
    }
};

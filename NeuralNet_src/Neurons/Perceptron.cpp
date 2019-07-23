/***************************************************************************//**
 * @file     Perceptron.cpp
 * @date     28 Mar 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Neurons
 * @brief    This class is responsable for model an Perceptron Neuron
 ******************************************************************************/

#include "Perceptron.h"

// Perceptron Constructor
Perceptron::Perceptron(int dimin, double *weights, bool bias, double sigma) {

    // Random number generator with mean 0. and std sigma.
    std::random_device rd_generator;
    std::normal_distribution <double> nm_distr(0., sigma);

    _dimin = dimin;
    _bias = bias;

    if (bias) {
        _wgtnumber = _dimin + 1;
    } else {
        _wgtnumber = _dimin;
    }

    _weights = new double [_wgtnumber];

    if (weights) {
        for (int wgt_=0; wgt_<_wgtnumber; wgt_++) {
            _weights[wgt_] = weights[wgt_];
        }
    }
    else {

        for (int wgt_=0; wgt_<_wgtnumber; wgt_++) {
            // Genereting a gaussian random number
            _weights[wgt_] = nm_distr(rd_generator);
        }
    }

};

// ~Perceptron Desctructor
Perceptron::~Perceptron(void) {
    delete _weights;
};

// update_weights
void Perceptron::update_weights(double *weights){
    cblas_dcopy(_wgtnumber, weights, 1, _weights, 1);
};

// response
double Perceptron::response(double *xin){

    double dotprod;

    // Performing dot product between xin and weights
    dotprod = cblas_ddot(_wgtnumber, xin, 1, _weights, 1);

    return ActvF(dotprod);
};

// dresp_du
double Perceptron::dresp_du(double *xin){

    double dotprod;

    // Performing dot product between xin and weights
    dotprod = cblas_ddot(_wgtnumber, xin, 1, _weights, 1);

    return dActvF_du(dotprod);
};

/***************************************************************************//**
 * @file     HyperbolicTangentPerceptron.h
 * @date     05 Apr 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Neurons
 * @brief    Class that models the perceptron neuron with the HyperbolicTangent
 *           activation function.
 ******************************************************************************/

#ifndef HYPERBOLICTANGENTPERCEPTRON_H
#define HYPERBOLICTANGENTPERCEPTRON_H

#include "Perceptron.h"
#include <cmath>

/***************************************************************************//**
 * @class HyperbolicTangentPerceptron
 * @brief Class that models the perceptron neuron with the HyperbolicTangent
 *        activation function.
 ******************************************************************************/
class HyperbolicTangentPerceptron : public Perceptron {

public:

    /*************************************************************************//**
     * Constructs a HyperbolicTangentPerceptron object
     ****************************************************************************/

    // HyperbolicTangentPerceptron Constructor
    /*************************************************************************//**
     * @brief Constructor of HyperbolicTangentPerceptron neuron class. If the neuron has the bias term,
     *        the last parameter of the input must be -1 and the last weight is the bias.
     *
     * @param   dimin   Neuron input dimension without the bias consideration
     * @param  weights  Vector of inicial weights for neuron inputs
     * @param   bias    Boolean for including the bias term for the neuron
     * @param  sigma    Standard deviation for neuron initialization
     * @param   beta    Slope of the HyperbolicTangent function
     ****************************************************************************/
    HyperbolicTangentPerceptron(int dimin, double *weights=NULL, bool bias=true, double sigma=1.0, double beta=1.0)
            : Perceptron(dimin, weights, bias, sigma) { _beta = beta; };

    // ~HyperbolicTangentPerceptron Desctructor
    virtual ~HyperbolicTangentPerceptron(void) {};     // Default destructor

    // HyperbolicTangentPerceptron Cloning
    virtual HyperbolicTangentPerceptron * Clone(){
            return new HyperbolicTangentPerceptron(_dimin, _weights, _bias, 1.0, _beta);};

    // ActvF
    /*************************************************************************//**
     * @brief Neuron activation function.
     *
     * @param    u    Input for funtion evaluation
     ****************************************************************************/
    double virtual ActvF(double u);

    // dActvF_du
    /*************************************************************************//**
     * @brief Derivative of the activation function in the u point
     *
     * @param      u      Input for funtion derivative evaluation
     ****************************************************************************/
    double virtual dActvF_du(double u);

protected:
    double _beta;

private:

};

#endif


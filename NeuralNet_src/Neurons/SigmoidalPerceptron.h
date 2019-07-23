/***************************************************************************//**
 * @file     SigmoidalPerceptron.h
 * @date     05 Apr 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Neurons
 * @brief    Class that models the perceptron neuron with the Sigmoidal
 *           activation function.
 ******************************************************************************/

#ifndef SIGMOIDALPERCEPTRON_H
#define SIGMOIDALPERCEPTRON_H

#include "Perceptron.h"
#include <cmath>

/***************************************************************************//**
 * @class SigmoidalPerceptron
 * @brief Class that models the perceptron neuron with the Sigmoidal
 *        activation function.
 ******************************************************************************/
class SigmoidalPerceptron : public Perceptron{

public:

    /*************************************************************************//**
     * Constructs a SigmoidalPerceptron object
     ****************************************************************************/

    // SigmoidalPerceptron Constructor
    /*************************************************************************//**
     * @brief Constructor of SigmoidalPerceptron neuron class. If the neuron has the bias term,
     *        the last parameter of the input must be -1 and the last weight is the bias.
     *
     * @param   dimin   Neuron input dimension without the bias consideration
     * @param  weights  Vector of inicial weights for neuron inputs
     * @param   bias    Boolean for including the bias term for the neuron
     * @param  sigma    Standard deviation for neuron initialization
     * @param   beta    Slope of the HyperbolicTangent function
     ****************************************************************************/
    SigmoidalPerceptron(int dimin, double *weights=NULL, bool bias=true, double sigma=1.0, double beta=1.0)
            : Perceptron(dimin, weights, bias, sigma) { _beta = beta; };

    // ~SigmoidalPerceptron Desctructor
    virtual ~SigmoidalPerceptron(void) {};     // Default destructor

    // SigmoidalPerceptron Cloning
    virtual SigmoidalPerceptron * Clone(){
            return new SigmoidalPerceptron(_dimin, _weights, _bias, 1.0, _beta);};

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


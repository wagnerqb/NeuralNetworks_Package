/***************************************************************************//**
 * @file     SHLPerceptron.h
 * @date     05 Apr 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Neurons
 * @brief    Class that models the perceptron neuron with the Symmetric Hard Limiter
 *           activation function.
 ******************************************************************************/

#ifndef SHLPERCEPTRON_H
#define SHLPERCEPTRON_H

#include "Perceptron.h"
#include <iostream>

/***************************************************************************//**
 * @class SHLPerceptron
 * @brief Class that models the perceptron neuron with the Symmetric Hard Limiter
 *        activation function.
 ******************************************************************************/
class SHLPerceptron : public Perceptron{

public:

    /*************************************************************************//**
     * Constructs a SHLPerceptron object
     ****************************************************************************/

    // SHLPerceptron Constructor
    /*************************************************************************//**
     * @brief Constructor of SHLPerceptron neuron class. If the neuron has the bias term,
     *        the last parameter of the input must be -1 and the last weight is the bias.
     *
     * @param   dimin   Neuron input dimension without the bias consideration
     * @param  weights  Vector of inicial weights for neuron inputs
     * @param   bias    Boolean for including the bias term for the neuron
     * @param  sigma    Standard deviation for neuron initialization
     ****************************************************************************/
    SHLPerceptron(int dimin, double *weights=NULL, bool bias=true, double sigma=1.0)
            : Perceptron(dimin, weights, bias, sigma) {};

    // ~SHLPerceptron Desctructor
    virtual ~SHLPerceptron(void) {};     // Default destructor

    // SHLPerceptron Cloning
    virtual SHLPerceptron * Clone(){
            return new SHLPerceptron(_dimin, _weights, _bias);};

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

private:

};

#endif


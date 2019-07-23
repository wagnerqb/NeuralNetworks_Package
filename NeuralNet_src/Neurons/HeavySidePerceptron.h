/***************************************************************************//**
 * @file     HeavySidePerceptron.h
 * @date     05 Apr 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Neurons
 * @brief    Class that models the perceptron neuron with the Heavyside
 *           activation function.
 ******************************************************************************/

#ifndef HEAVYSIDEPERCEPTRON_H
#define HEAVYSIDEPERCEPTRON_H

#include "Perceptron.h"

/***************************************************************************//**
 * @class HeavySidePerceptron
 * @brief Class that models the perceptron neuron with the Heavyside
 *        activation function.
 ******************************************************************************/
class HeavySidePerceptron : public Perceptron{

public:

    /*************************************************************************//**
     * Constructs a HeavySidePerceptron object
     ****************************************************************************/

    // HeavySidePerceptron Constructor
    /*************************************************************************//**
     * @brief Constructor of HeavySidePerceptron neuron class. If the neuron has the bias term,
     *        the last parameter of the input must be -1 and the last weight is the bias.
     *
     * @param   dimin   Neuron input dimension without the bias consideration
     * @param  weights  Vector of inicial weights for neuron inputs
     * @param   bias    Boolean for including the bias term for the neuron
     * @param  sigma    Standard deviation for neuron initialization
     ****************************************************************************/
    HeavySidePerceptron(int dimin, double *weights=NULL, bool bias=true, double sigma=1.0)
            : Perceptron(dimin, weights, bias, sigma) {};

    // ~HeavySidePerceptron Desctructor
    virtual ~HeavySidePerceptron(void) {};     // Default destructor

    // HeavySidePerceptron Cloning
    virtual HeavySidePerceptron * Clone(){
            return new HeavySidePerceptron(_dimin, _weights, _bias);};

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


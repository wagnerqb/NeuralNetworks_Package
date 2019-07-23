/***************************************************************************//**
 * @file     SymmetricLinearPerceptron.h
 * @date     05 Apr 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Neurons
 * @brief    Class that models the perceptron neuron with the Symmetric Linear
 *           activation function.
 ******************************************************************************/

#ifndef SYMMETRICLINEARPERCEPTRON_H
#define SYMMETRICLINEARPERCEPTRON_H

#include "Perceptron.h"
#include <cmath>

/***************************************************************************//**
 * @class SymmetricLinearPerceptron
 * @brief Class that models the perceptron neuron with the Symmetric Linear
 *        activation function.
 ******************************************************************************/
class SymmetricLinearPerceptron : public Perceptron{

public:

    /*************************************************************************//**
     * Constructs a SymmetricLinearPerceptron object
     ****************************************************************************/

    // SymmetricLinearPerceptron Constructor
    /*************************************************************************//**
     * @brief Constructor of SymmetricLinearPerceptron neuron class. If the neuron has the bias term,
     *        the last parameter of the input must be -1 and the last weight is the bias.
     *
     * @param   dimin   Neuron input dimension without the bias consideration
     * @param  weights  Vector of inicial weights for neuron inputs
     * @param   bias    Boolean for including the bias term for the neuron
     * @param  sigma    Standard deviation for neuron initialization
     * @param  a_slope  Constant value of 0 slope
     ****************************************************************************/
    SymmetricLinearPerceptron(int dimin, double *weights=NULL, bool bias=true, double sigma=1.0, double a_slope=1.0)
            : Perceptron(dimin, weights, bias, sigma) { _a_slope = a_slope; };

    // ~SymmetricLinearPerceptron Desctructor
    virtual ~SymmetricLinearPerceptron(void) {};     // Default destructor

    // SymmetricLinearPerceptron Cloning
    virtual SymmetricLinearPerceptron * Clone(){
            return new SymmetricLinearPerceptron(_dimin, _weights, _bias, 0., _a_slope);};

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
    double _a_slope;

private:

};

#endif


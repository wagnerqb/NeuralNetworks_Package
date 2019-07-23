/***************************************************************************//**
 * @file     Perceptron.h
 * @date     28 Mar 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Neurons
 * @brief    This class is responsable for model an Perceptron Neuron
 ******************************************************************************/

#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "mkl.h"
#include <random>

/***************************************************************************//**
 * @class Perceptron
 * @brief This class is responsable for model an Perceptron Neuron.
 * @ingroup Neurons
 ******************************************************************************/
class Perceptron {

public:

    /*************************************************************************//**
     * Constructs a Perceptron object
     ****************************************************************************/

    // Perceptron Constructor
    /*************************************************************************//**
     * @brief Constructor of perceptron neuron class. If the neuron has the bias term,
     *        the last parameter of the input must be -1 and the last weight is the bias.
     *
     * @param   dimin   Neuron input dimension without the bias consideration
     * @param  weights  Vector of inicial weights for neuron inputs
     * @param   bias    Boolean for including the bias term for the neuron
     * @param  sigma    Standard deviation for neuron initialization
     ****************************************************************************/
    Perceptron(int dimin, double *weights=NULL, bool bias=true, double sigma=1.0);

    // ~Perceptron Desctructor
    virtual ~Perceptron(void);     // Default destructor

    // Perceptron Cloning
    virtual Perceptron * Clone() {};

    // ActvF
    /*************************************************************************//**
     * @brief Neuron activation function.
     *
     * @param    u    Input for funtion evaluation
     ****************************************************************************/
    double virtual ActvF(double u) {return 0.;};

    // dActvF_du
    /*************************************************************************//**
     * @brief Derivative of the activation function in the u point
     *
     * @param      u      Input for funtion derivative evaluation
     ****************************************************************************/
    double virtual dActvF_du(double u) {
        double eps_ = 1.0e-5;
        return (ActvF(u+eps_) - ActvF(u))/eps_;
    };

    // update_weights
    /*************************************************************************//**
     * @brief Function that updates the neuron weights.
     *
     * @param  weights  Vector of inicial weights for neuron inputs
     ****************************************************************************/
    void update_weights(double *weights);

    // response
    /*************************************************************************//**
     * @brief Response of the neuron subject of the inlet xin. If the neuron has the bias term,
     *        the first parameter of the input must be -1 and the first weight is the bias.
     *
     * @param  xin  neuron inlet
     ****************************************************************************/
    double response(double *xin);

    // dresp_du
    /*************************************************************************//**
     * @brief Returns the derivative of the response related of the inlet weighted variable.
     *        If the neuron has the bias term, the first parameter of the input must be -1
     *        and the first weight is the bias.
     *
     * @param  xin  neuron inlet
     ****************************************************************************/
    double dresp_du(double *xin);

    // Getters ans Setters
    inline double * weights() { return _weights; }


protected:

    int _dimin;        // Neuron input dimension without the bias consideration;
    bool _bias;        // Flag showing if bias parameter is being considered
    int _wgtnumber;    // Number of weights in Neuron including bias if exists;
    double *_weights;  // Vector of weights inside the Neuron

private:

};

#endif


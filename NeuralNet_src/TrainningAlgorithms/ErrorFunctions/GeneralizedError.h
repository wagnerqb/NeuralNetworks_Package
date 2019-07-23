/***************************************************************************//**
 * @file     GeneralizedError.h
 * @date     04 Jun 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup ErrorFunctions
 * @brief    Class responsible for the error calculus between the trainning
 *           data and the network response using the generalized error model.
 ******************************************************************************/

#ifndef GENERALIZEDERROR_H
#define GENERALIZEDERROR_H

#include <vector>
#include <iostream>
#include <cmath>
#include "mkl.h"
#include "SumSquaresMean.h"

/***************************************************************************//**
 * @class GeneralizedError
 * @brief Class responsible for the error calculus between the trainning
 *        data and the network response using the generalized error model.
 * @ingroup ErrorFunctions
 ******************************************************************************/
class GeneralizedError : public SumSquaresMean {

public:

    /*************************************************************************//**
     * Constructs a GeneralizedError object
     ****************************************************************************/

    // GeneralizedError Constructor
    /*************************************************************************//**
     * @brief Constructor of GeneralizedError error function class.
     *
     * @param   layers   Number of layers in the MLP Networks
     * @param   wgtcfg   Array with number of weights in each neuron layer
     * @param  layercfg  Array with number of neurons in each layer
     * @param    _lbd    Regularization parameter
     * @param   _beta    Least Squares Error Weight
     ****************************************************************************/
    GeneralizedError(int layers, int * wgtcfg, int * layercfg, double lbd=0.0, double beta=1.0);

    // ~GeneralizedError Desctructor
    virtual ~GeneralizedError(void);     // Default destructor

    // Network_Error
    /*************************************************************************//**
     * @brief Function that calculates the error between the network output and
     *        the vector of expected responses.
     *
     * @param  netresponse   Array of network responses
     * @param  expresponse   Array of expected responses
     ****************************************************************************/
    double virtual Network_Error(double *netresponse, double *expresponse);

    // dNeterror_dResp
    /*************************************************************************//**
     * @brief Function that calculates the derivative of total error
     *        related with the network response.
     *
     * @param  netresponse  Array of network responses
     * @param  expresponse  Array of expected responses
     ****************************************************************************/
    double virtual * dNeterror_dResp(double *netresponse, double *expresponse);

protected:

    double _beta;            // Least Squares weight
    double *_crss_entrp;     // Vector to stores the cross entropy error
    double *_dcrsetp_dresp;  // Vector with the derivative of the cross entropy error

private:

};

#endif


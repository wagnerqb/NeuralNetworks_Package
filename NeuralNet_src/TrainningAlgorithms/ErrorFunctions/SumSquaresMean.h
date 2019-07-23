/***************************************************************************//**
 * @file     SumSquaresMean.h
 * @date     03 May 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup ErrorFunctions
 * @brief    Class responsible for the error calculus between the trainning
 *           data and the network response using the sum of squares mean model.
 ******************************************************************************/

#ifndef SUMSQUARESMEAN_H
#define SUMSQUARESMEAN_H

#include <vector>
#include <iostream>
#include "mkl.h"
#include "ErrorFunctions_Model.h"

/***************************************************************************//**
 * @class SumSquaresMean
 * @brief Class responsible for the error calculus between the trainning
 *        data and the network response using the sum of squares mean model.
 * @ingroup ErrorFunctions
 ******************************************************************************/
class SumSquaresMean : public ErrorFunctions_Model {

public:

    /*************************************************************************//**
     * Constructs a SumSquaresMean object
     ****************************************************************************/

    // SumSquaresMean Constructor
    /*************************************************************************//**
     * @brief Constructor of SumSquaresMean error function class.
     *
     * @param   layers   Number of layers in the MLP Networks
     * @param   wgtcfg   Array with number of weights in each neuron layer
     * @param  layercfg  Array with number of neurons in each layer
     * @param    _lbd    Regularization parameter
     ****************************************************************************/
    SumSquaresMean(int layers, int * wgtcfg, int * layercfg, double lbd=0.0);

    // ~SumSquaresMean Desctructor
    virtual ~SumSquaresMean(void);     // Default destructor

    // Error
    /*************************************************************************//**
     * @brief Function that calculates the error between the network output and
     *        the vector of expected responses and add the regularization term.
     *
     * @param   netresponse  Array of network responses
     * @param   expresponse  Array of expected responses
     * @param  net_wghtcont  Container storing the weights for all neurons in all layers
     ****************************************************************************/
    double virtual Error(double *netresponse, double *expresponse, std::vector < std::vector <double *> > net_wghtcont);

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

    // Regularization
    /*************************************************************************//**
     * @brief Function that calculates the regularization part of error.
     *
     * @param  net_wghtcont  Container storing the weights for all neurons in all layers
     ****************************************************************************/
    double virtual Regularization(std::vector < std::vector <double *> > net_wghtcont);

    // dRegularization_dWeight
    /*************************************************************************//**
     * @brief Function that calculates the derivative of regularization error
     *        related to all neuron weights.
     *
     * @param  net_wghtcont  Container storing the weights for all neurons in all layers
     ****************************************************************************/
    void virtual Calculate_dRegularization_dWeight(std::vector < std::vector <double *> > net_wghtcont);

protected:

    double *_diffvec;       // Vector to stores the difference between network output and
                            // expected response

private:

};

#endif


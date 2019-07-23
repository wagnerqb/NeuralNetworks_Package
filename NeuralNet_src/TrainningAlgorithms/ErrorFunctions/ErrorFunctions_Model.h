/***************************************************************************//**
 * @file     ErrorFunctions_Model.h
 * @date     06 Jun 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup ErrorFunctions
 * @brief    Class responsible for the model of error calculus between the trainning
 *           data and the network response, used for polymorfism purposes.
 ******************************************************************************/

#ifndef ERRORFUNCTIONS_MODEL_H
#define ERRORFUNCTIONS_MODEL_H

#include <vector>
#include <iostream>
#include "mkl.h"

/***************************************************************************//**
 * @class ErrorFunctions_Model
 * @brief Class responsible for the model of error calculus between the trainning
 *        data and the network response, used for polymorfism purposes.
 * @ingroup ErrorFunctions
 ******************************************************************************/
class ErrorFunctions_Model {

public:

    /*************************************************************************//**
     * Constructs a ErrorFunctions_Model object
     ****************************************************************************/

    // ErrorFunctions_Model Constructor
    /*************************************************************************//**
     * @brief Constructor of ErrorFunctions_Model error function class.
     ****************************************************************************/
    ErrorFunctions_Model() {};

    // ~ErrorFunctions_Model Desctructor
    virtual ~ErrorFunctions_Model(void) {};     // Default destructor

    // Error
    /*************************************************************************//**
     * @brief Function that calculates the error between the network output and
     *        the vector of expected responses and add the regularization term.
     *
     * @param   netresponse  Array of network responses
     * @param   expresponse  Array of expected responses
     * @param  net_wghtcont  Container storing the weights for all neurons in all layers
     ****************************************************************************/
    double virtual Error(double *netresponse, double *expresponse, std::vector < std::vector <double *> > net_wghtcont) = 0;

    // Network_Error
    /*************************************************************************//**
     * @brief Function that calculates the error between the network output and
     *        the vector of expected responses.
     *
     * @param  netresponse   Array of network responses
     * @param  expresponse   Array of expected responses
     ****************************************************************************/
    double virtual Network_Error(double *netresponse, double *expresponse) = 0;

    // dNeterror_dResp
    /*************************************************************************//**
     * @brief Function that calculates the derivative of total error
     *        related with the network response.
     *
     * @param  netresponse  Array of network responses
     * @param  expresponse  Array of expected responses
     ****************************************************************************/
    double virtual * dNeterror_dResp(double *netresponse, double *expresponse) = 0;

    // Regularization
    /*************************************************************************//**
     * @brief Function that calculates the regularization part of error.
     *
     * @param  net_wghtcont  Container storing the weights for all neurons in all layers
     ****************************************************************************/
    double virtual Regularization(std::vector < std::vector <double *> > net_wghtcont) = 0;

    // dRegularization_dWeight
    /*************************************************************************//**
     * @brief Function that calculates the derivative of regularization error
     *        related to all neuron weights.
     *
     * @param  net_wghtcont  Container storing the weights for all neurons in all layers
     ****************************************************************************/
    void virtual Calculate_dRegularization_dWeight(std::vector < std::vector <double *> > net_wghtcont) = 0;

    // Getters ans Setters
    std::vector < std::vector <double *> > dreg_dwgt_cont() { return _dreg_dwgt_cont; };

protected:

    double _lbd;            // Regularization parameter weight
    int _outnetnbr;         // Number of outputs in network
    int _totalwgtnbr;       // Total number of weights in network
    int _layers;            // Number of layers in network;
    int *_wgtcfg;           // List of weights number in each layer;
    int *_layercfg;         // List of neurons in each layer;
    double *_dNeterr_dresp; // Vector storing the derivative of network error by each expected response
    std::vector < std::vector <double *> > _dreg_dwgt_cont;    // Container for store the
                            // derivative of regularization error related to all weights (speedup purpose);

private:

};

#endif


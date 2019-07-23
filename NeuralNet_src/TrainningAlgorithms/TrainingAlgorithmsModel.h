/***************************************************************************//**
 * @file     TrainingAlgorithmsModel.h
 * @date     04 Jun 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Networks
 * @brief    This class is responsable for create a model of training classes.
 ******************************************************************************/

#ifndef TRAININGALGORITHMSMODEL_H
#define TRAININGALGORITHMSMODEL_H

#include <vector>
#include <iostream>
#include "MLP_Network.h"
#include "ErrorFunctions_Model.h"
#include "SumSquaresMean.h"
#include "GeneralizedError.h"

/***************************************************************************//**
 * @class TrainingAlgorithmsModel
 * @brief This class is responsable for create a model of training classes.
 * @ingroup TrainingAlgorithms
 ******************************************************************************/
class TrainingAlgorithmsModel {

public:

    /*************************************************************************//**
     * Constructs a TrainingAlgorithmsModel object
     ****************************************************************************/

    // TrainingAlgorithmsModel Constructor
    /*************************************************************************//**
     * @brief Constructor of TrainingAlgorithmsModel class.
     *
     * @param   network  Pointer to the network structure that will be trainned
     * @param    beta    Least Squares weight factor
     * @param    lbd     Regularization parameter
     ****************************************************************************/
    TrainingAlgorithmsModel(MLP_Network *network, double beta=1.0, double lbd=0.0);

    // ~TrainingAlgorithmsModel Desctructor
    virtual ~TrainingAlgorithmsModel(void);     // Default destructor

    // Clean_dErrordWeight
    /*************************************************************************//**
     * @brief Function that clean the _derrordweight container.
     ****************************************************************************/
    void Clean_dErrordWeight(void);

    // Update_Error_Derivatives
    /*************************************************************************//**
     * @brief Function that updates the error and the derivatives of the error
     *        related with each neuron weight, and stores the result into a
     *        specific container.
     *
     * @param  evaldata  list of Data that will be evaluated by the network.
     *                   This data is in a list where the last value is the
     *                   expected result;
     * @param    itbgn   Index with the first element of evaldat that will
     *                   be evaluated;
     * @param    itend   Index with the last element of evaldat that will
     *                   be evaluated;
     ****************************************************************************/
    void Update_Error_Derivatives(std::vector < double * > evaldata, int itbgn, int itend);

    /*************************************************************************//**
     * @brief Function reponsable for training the network with the training data.
     *
     * @param  trainingdata  Trainingdata vector, where each parameter is an array
     *                       with the trainning data and the last element are the
     *                       expected values;
     * @param      eta       Learning rate;
     * @param   max_epoch    Max number of iterations;
     * @param      tol       Tolerance of network;

     * @param   derror_tol   Tolerance for error variation of the network;
     * @param  batch_ratio   Ratio of the data used per optimization cycle;
     * @param   reportfreq   Frequence of reported data;
     * @param   early_stop   Activate the early stop if the validation data
                             error stops the convergence.
     ****************************************************************************/
    void virtual Training(std::vector < double * > trainingdata, double eta=0.0025, int max_epoch=10000,
                          double tol=1.e-6, double derror_tol=1.e-8, double batch_ratio=1.0, int reportfreq=100,
                          bool reporttime=true, bool early_stop=false) = 0;

    // Getters ans Setters
    inline std::vector < std::vector < double * > >  derrordweight() { return _derrordweight; };

protected:

    MLP_Network *_network;     // Network that will be trainned
    ErrorFunctions_Model *_errorfunction; // Function for error evaluation
    std::vector < std::vector < double * > > _derrordweight;    // Container for store the
                               // derivative of error for each weights in network
    double _globalerror;       // Stores the global error in a network evaluation
    std::vector < double * > _derrordu;    // Derivative of the error for each neuron input
                               // (partial calculus purpose)

private:

};

#endif


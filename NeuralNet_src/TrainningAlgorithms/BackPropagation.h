/***************************************************************************//**
 * @file     BackPropagation.h
 * @date     05 Jun 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Networks
 * @brief    This class is responsable for create a training classes based
 *           on the back propagation algorithm.
 ******************************************************************************/

#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include <ctime>
#include <iomanip>
#include "TrainingAlgorithmsModel.h"

/***************************************************************************//**
 * @class BackPropagation
 * @brief This class is responsable for create a training classes based
 *        on the back propagation algorithm.
 * @ingroup TrainingAlgorithms
 ******************************************************************************/
class BackPropagation : public TrainingAlgorithmsModel {

public:

    /*************************************************************************//**
     * Constructs a BackPropagation object
     ****************************************************************************/

    // BackPropagation Constructor
    /*************************************************************************//**
     * @brief Constructor of BackPropagation class.
     *
     * @param   network  Pointer to the network structure that will be trainned
     * @param    beta    Least Squares weight factor
     * @param    lbd     Regularization parameter
     ****************************************************************************/
    BackPropagation(MLP_Network * network, double beta=1.0, double lbd=0.0)
                    : TrainingAlgorithmsModel(network, beta, lbd) {};

    // ~BackPropagation Desctructor
    virtual ~BackPropagation(void) {};     // Default destructor

    // Training
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
                          bool reporttime=true, bool early_stop=false);


protected:

private:

};

#endif


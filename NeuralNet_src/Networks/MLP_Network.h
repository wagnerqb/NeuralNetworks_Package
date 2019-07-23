/***************************************************************************//**
 * @file     MLP_Network.h
 * @date     01 May 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Networks
 * @brief    This class is responsable for model a MLP (Multi Layer Perceptron)
             network.
 ******************************************************************************/

#ifndef MLP_NETWORK_H
#define MLP_NETWORK_H

#include <vector>
#include <iostream>
#include "Perceptron.h"
#include "mkl.h"

/***************************************************************************//**
 * @class MLP_Network
 * @brief This class is responsable for model a Multilayer Perceptron Network.
 * @ingroup Networks
 ******************************************************************************/
class MLP_Network {

public:

    /*************************************************************************//**
     * Constructs a MLP_Network object
     ****************************************************************************/

    // MLP_Network Constructor
    /*************************************************************************//**
     * @brief Constructor of MLP_Network class. Each neuron inside this structure
     *        needs to be a perceptron type.
     *
     * @param   layers   Number of layers in the MLP Networks
     * @param  dimnetin  Network input dimension (excluding the bias term)
     * @param   biascfg  List for each neuron layer, showing if it has the bias term
     ****************************************************************************/
    MLP_Network(int layers, int dimnetin, bool *biascfg);

    // ~MLP_Network Desctructor
    virtual ~MLP_Network(void);     // Default destructor

    // Insert_Neuron
    /*************************************************************************//**
     * @brief Function that adds a new neuron in a specified layer.
     *
     * @param    nrn    Pointer to the neuron to be inserted
     * @param  lyr_ins  Layer where the Neuron will be inserted (counting starts in 0)
     ****************************************************************************/
    void Insert_Neuron(Perceptron *nrn, int lyr_ins);

    // Evaluate_Network
    /*************************************************************************//**
     * @brief Function that calculates all network reponse when subject to
     *        the inlet xin and stores inside the network structure.
     *
     * @param    xin    Netowork Input
     ****************************************************************************/
    void Evaluate_Network(double *xin);

    // Calculate_Network
    /*************************************************************************//**
     * @brief Function that calculates the response of the network and the derivatives
     *        of each activation function. Used for trainning purposes.
     *
     * @param    xin    Netowork Input
     ****************************************************************************/
    void Calculate_Network(double *xin);

    // Update_NeuralWeights_Container
    /*************************************************************************//**
     * @brief Function that updates the neural weight container with the weights
     *        of all neurons in problem.
     ****************************************************************************/
    void Update_NeuralWeights_Container(void);

    // Response
    /*************************************************************************//**
     * @brief Response of the network subject of the inlet xin.
     *
     * @param    xin    Netowork Input
     ****************************************************************************/
    double * Response(double *xin);

    // Getters ans Setters
    inline int layers() { return _layers; };
    inline int dimnetin() { return _dimnetin; };
    inline int * layercfg() { return _layercfg; };
    inline int * wgtcfg() { return _wgtcfg; };
    inline bool * biascfg() { return _biascfg; };
    // Update the weigths of neuron nrn in layer lyr
    inline void UpdateNeuronWeight(int lyr, int nrn, double * wgts) { _network[lyr][nrn]->update_weights(wgts); };

    // This getters below are dangerous because the network should be calculated first
    inline double ** rsltcont() { return _rsltcont; };
    inline double ** actvfdercont() { return _actvfdercont; };
    inline double * iptcont() { return _iptcont; };
    inline std::vector < std::vector <double *> > wghtcont() { return _wghtcont; };

protected:

    int _dimnetin;          // Network input dimension;
    int _layers;            // Number of layers in network;
    bool *_biascfg;         // List for bias term configuration for each layer;
    std::vector < std::vector <Perceptron *> > _network;  // Vector of vectors of
                            // perceptron neuron pointers;
    int *_layercfg;         // List of neurons in each layer;
    int *_wgtcfg;           // List of weights number in each layer;
    double *_iptcont;       // Container for store the input used in the last
                            // response calculus (speedup purpose);
    double **_rsltcont;     // Container for store the result calculated by each neuron
                            // in each layer (speedup purpose);
    double **_actvfdercont; // Container for store the activation function derivative
                            // calculated by each neuron in all layers (speedup purpose);
    std::vector < std::vector <double *> > _wghtcont;    // Container for store the
                            // weights for all neurons in all layers (speedup purpose);

private:

};

#endif


/***************************************************************************//**
 * @file     MLP_Network.cpp
 * @date     01 May 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Networks
 * @brief    This class is responsable for model a MLP (Multi Layer Perceptron)
             network.
 ******************************************************************************/

#include "MLP_Network.h"

// MLP_Network Constructor
MLP_Network::MLP_Network(int layers, int dimnetin, bool *biascfg) {

    _layers = layers;       // Number of layers in network;
    _dimnetin = dimnetin;   // Network input dimension;

    // Initiating the bias configuration list
    _biascfg = new bool [_layers];
    for (int lyr_=0; lyr_<_layers; lyr_++) {
        _biascfg[lyr_] = biascfg[lyr_];
    }

    // Initiating the perceptron network list and the layer configuration array
    _layercfg = new int [_layers];
    for (int lyr_=0; lyr_<_layers; lyr_++) {
        _layercfg[lyr_] = 0;
        std::vector <Perceptron *> nrn_vec_; // Creating an empty neurons pointer vector;
        _network.push_back(nrn_vec_);
    }

    // Initiating the weight configuration list
    _wgtcfg = new int [_layers];
    for (int lyr_=0; lyr_<_layers; lyr_++) {
        _wgtcfg[lyr_] = 0;

        // If the layer has bias
        if (_biascfg[lyr_]) {
            _wgtcfg[lyr_]++;
        }
    }
    // Initiating the first layer with the input dimension
    _wgtcfg[0] += _dimnetin;

    // Initiating the input container
    if (_biascfg[0]) {
        // If first layer has the bias term
        _iptcont = new double[_dimnetin+1];
        _iptcont[_dimnetin] = -1.;
    } else {
        _iptcont = new double[_dimnetin];
    }
    for (int inpt_=0; inpt_<_dimnetin; inpt_++) {
        _iptcont[inpt_] = 0;
    }

    // Initiating the results container
    _rsltcont = new double * [_layers];
    for (int lyr_=0; lyr_<_layers; lyr_++) {
        _rsltcont[lyr_] = nullptr;
    }

    // Initiating the weights container
    for (int lyr_=0; lyr_<_layers; lyr_++) {
        std::vector <double *> nrn_wgt_; // Creating an empty neurons pointer vector;
        _wghtcont.push_back(nrn_wgt_);
    }

    // Initiating the activation function derivative container
    _actvfdercont = new double * [_layers];
    for (int lyr_=0; lyr_<_layers; lyr_++) {
        _actvfdercont[lyr_] = nullptr;
    }

};

// ~MLP_Network Desctructor
MLP_Network::~MLP_Network(void) {

    // Destructing the bias configuration term
    delete _biascfg;

    // Destructing the results container
    for (int lyr_=0; lyr_<_layers; lyr_++) {
        delete _rsltcont[lyr_];
    }
    delete _rsltcont;

    // Destructing the activation function derivative container
    for (int lyr_=0; lyr_<_layers; lyr_++) {
        delete _actvfdercont[lyr_];
    }
    delete _actvfdercont;

    // Destructing the perceptron network list
    for (int lyr_=0; lyr_<_layers; lyr_++) {
        for (int nrn_=0; nrn_<_layercfg[lyr_]; nrn_++) {
            delete _network[lyr_][nrn_];
        }
    }

    // Destructing the weights container vector
    for (int lyr_=0; lyr_<_layers; lyr_++) {
        for (int nrn_=0; nrn_<_layercfg[lyr_]; nrn_++) {
            delete _wghtcont[lyr_][nrn_];
        }
    }

    // Destructing the Layer configuration list
    delete _layercfg;

    // Destructing the weight configuration list
    delete _wgtcfg;

    // Destructing the input container
    delete _iptcont;

};

// Insert_Neuron
void MLP_Network::Insert_Neuron(Perceptron *nrn, int lyr_ins) {
    double *aux_insert=nullptr;

    _network[lyr_ins].push_back(nrn->Clone());
    _layercfg[lyr_ins]++;

    // Updating the results container size
        // Step 1: Cloning the actual result list
        // Checking if the next layer does not have the bias term
        // or if it is the last layer

        if ((lyr_ins != _layers-1) and (_biascfg[lyr_ins+1])) {
            //  The next layer has the bias term, so it is include the -1
                aux_insert = new double [_layercfg[lyr_ins]+1];
                for (int nrn_=0; nrn_ < _layercfg[lyr_ins]; nrn_++) {
                    aux_insert[nrn_] = 0.;
                }
                aux_insert[_layercfg[lyr_ins]] = -1.;
        } else {
                aux_insert = new double [_layercfg[lyr_ins]];
                for (int nrn_=0; (nrn_<_layercfg[lyr_ins]); nrn_++) {
                    aux_insert[nrn_] = 0.;
                }
        }

        // Step 2: Cleaning the actual result list
        delete _rsltcont[lyr_ins];

        // Step 3: Assigning the new list
        _rsltcont[lyr_ins] = aux_insert;
        aux_insert = nullptr;

    // Updating the weight configuration list
    if (lyr_ins != _layers-1) {
        _wgtcfg[lyr_ins+1]++;
    }

    // Updating the weights container vector
        // Step 1: Inserting a new array inside de vector
        _wghtcont[lyr_ins].push_back(new double [_wgtcfg[lyr_ins]]);
        for (int wgt_=0; wgt_<_wgtcfg[lyr_ins]; wgt_++) {
            int nrn_ = _wghtcont[lyr_ins].size();
            _wghtcont[lyr_ins][nrn_-1][wgt_] = 0.;
        }

        // Step 2: Updating the arrays already inserted in the next layer
        if (lyr_ins != _layers-1) {
            for (int nrn_=0; nrn_ < _layercfg[lyr_ins+1]; nrn_++) {
                delete _wghtcont[lyr_ins+1][nrn_];
                _wghtcont[lyr_ins+1][nrn_] = new double [_wgtcfg[lyr_ins+1]];
               for (int wgt_=0; wgt_<_wgtcfg[lyr_ins+1]; wgt_++) {
                    _wghtcont[lyr_ins+1][nrn_][wgt_] = 0.;
                }

            }
        }

    // Updating the activation activation function derivative container
        // Step 1: Cloning the actual activation derivative list

        aux_insert = new double [_layercfg[lyr_ins]];
        for (int nrn_=0; (nrn_<_layercfg[lyr_ins]); nrn_++) {
            aux_insert[nrn_] = 0.;
        }

        // Step 2: Cleaning the actual activation function derivative list
        delete _actvfdercont[lyr_ins];

        // Step 3: Assigning the new list
        _actvfdercont[lyr_ins] = aux_insert;
        aux_insert = nullptr;

};

// Evaluate_Network
void MLP_Network::Evaluate_Network(double *xin){

    // Step 1 - Updating The input container
    for (int inpt_=0; inpt_<_dimnetin; inpt_++) {
        _iptcont[inpt_] = xin[inpt_];
    }

    // Step 2 - Evaluating the first layer
    for (int nrn_=0; nrn_ < _layercfg[0]; nrn_++) {
        _rsltcont[0][nrn_] = _network[0][nrn_]->response(_iptcont);
    }

    // Step 3 - Evaluating the hidden layers
    for (int lyr_=1; lyr_<_layers; lyr_++){
        for (int nrn_=0; nrn_ < _layercfg[lyr_]; nrn_++) {
            _rsltcont[lyr_][nrn_] = _network[lyr_][nrn_]->response(_rsltcont[lyr_-1]);
        }
    }

};

// Calculate_Network
void MLP_Network::Calculate_Network(double *xin){

    // 1 - Updating The results container
    Evaluate_Network(xin);

    // 2 - Updating the derivative of the activation function container
        // Evaluation of the first layer
        for (int nrn_=0; nrn_<_layercfg[0]; nrn_++) {
            _actvfdercont[0][nrn_] = _network[0][nrn_]->dresp_du(_iptcont);
        }

        // Evaluation of the hidden layers
        for (int lyr_=1; lyr_<_layers; lyr_++){
            for (int nrn_=0; nrn_ < _layercfg[lyr_]; nrn_++) {
                _actvfdercont[lyr_][nrn_] = _network[lyr_][nrn_]->dresp_du(_rsltcont[lyr_-1]);
            }
        }

    // 3 - Updating the neuron weights container
    Update_NeuralWeights_Container();

};

// Update_NeuralWeights_Container
void MLP_Network::Update_NeuralWeights_Container(void) {

    for (int lyr_=0; lyr_<_layers; lyr_++) {
        for (int nrn_=0; nrn_ < _layercfg[lyr_]; nrn_++) {
            cblas_dcopy(_wgtcfg[lyr_], _network[lyr_][nrn_]->weights(), 1, _wghtcont[lyr_][nrn_], 1);
        }
    }

};

// Response
double * MLP_Network::Response(double *xin){
        Evaluate_Network(xin);
        return _rsltcont[_layers-1];
};

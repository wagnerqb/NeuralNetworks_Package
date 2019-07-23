/***************************************************************************//**
 * @file     TrainingAlgorithmsModel.cpp
 * @date     04 Jun 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Networks
 * @brief    This class is responsable for create a model of training classes.
 ******************************************************************************/

#include "TrainingAlgorithmsModel.h"

// TrainingAlgorithmsModel Constructor
TrainingAlgorithmsModel::TrainingAlgorithmsModel(MLP_Network *network, double beta, double lbd){

    int *lyr_cfg;  // Auxiliar for layer configuration vector
    int nrnipt;       // Auxiliar for neuron input count

    _network = network;
    // Error function
    if (beta == 1.0) {
        // Sum of Squares Mean Error
        _errorfunction = new SumSquaresMean(_network->layers(), _network->wgtcfg(), _network->layercfg(), lbd);
    } else {
        _errorfunction = new GeneralizedError(_network->layers(), _network->wgtcfg(), _network->layercfg(), lbd, beta);
    }

    // Initiating the global error
    _globalerror = 0.;

    // Initiating the derivative of error related with each neuron weight container
    lyr_cfg = _network->layercfg();
    for (int lyr_=0; lyr_ < _network->layers(); lyr_++) {

        // Initiating _derrordu for each layer
        _derrordu.push_back( new double [lyr_cfg[lyr_]] );

        // Initiating _derrordweight for each neuron
        std::vector < double * > dedwgt_; // Creating an empty double pointer vector;
        _derrordweight.push_back(dedwgt_);
        for (int nrn_=0; nrn_ < lyr_cfg[lyr_]; nrn_++) {

            // Initiating values for _derrordu
            _derrordu[lyr_][nrn_] = 0.;

            // Counting the number of neuron inputs
            if (lyr_ == 0) {
                nrnipt = _network->dimnetin();  // First Layer
            } else {
                nrnipt = lyr_cfg[lyr_-1];      // Hidden Layers
            }

            // Check if the layer has the bias term
            if (_network->biascfg()[lyr_]) {
                _derrordweight[lyr_].push_back( new double [nrnipt + 1] );

                // Initiating values for _derrordweight
                for (int nipt_=0; nipt_ < nrnipt+1; nipt_++) {
                    _derrordweight[lyr_][nrn_][nipt_] = 0.;
                }

            } else {
                _derrordweight[lyr_].push_back( new double [nrnipt] );

                // Initiating values for _derrordweight
                for (int nipt_=0; nipt_ < nrnipt; nipt_++) {
                    _derrordweight[lyr_][nrn_][nipt_] = 0.;
                }
            }
        }
    }
};

// ~TrainingAlgorithmsModel Desctructor
TrainingAlgorithmsModel::~TrainingAlgorithmsModel(void){
    int *lyr_cfg;

    // Destructing _derrordu
    for (int lyr_=0; lyr_ < _network->layers(); lyr_++) {
        delete _derrordu[lyr_];
    }

    lyr_cfg = _network->layercfg();
    // Destructing _derrordweight
    for (int lyr_=0; lyr_ < _network->layers(); lyr_++) {
        for (int nrn_=0; nrn_ < lyr_cfg[lyr_]; nrn_++) {
            delete _derrordweight[lyr_][nrn_];
        }
    }

    delete _errorfunction;
};

// Clean_dErrordWeight
void TrainingAlgorithmsModel::Clean_dErrordWeight(void){
    int *lyr_cfg;  // Auxiliar for layer configuration vector
    int *wgt_cfg;  // Auxiliar for weight count vector
    int nrnipt;       // Auxiliar for neuron input count

    // Cleaning Global Error
    _globalerror = 0.;

    wgt_cfg = _network->wgtcfg();
    lyr_cfg = _network->layercfg();
    // Cleaning _derrordu and _derrordweight
    for (int lyr_=0; lyr_ < _network->layers(); lyr_++) {
        for (int nrn_=0; nrn_ < lyr_cfg[lyr_]; nrn_++) {

            // Initiating values for _derrordu
            _derrordu[lyr_][nrn_] = 0.;

            // Initiating values for _derrordweight
            for (int wgt_=0; wgt_ < wgt_cfg[lyr_]; wgt_++) {
                _derrordweight[lyr_][nrn_][wgt_] = 0.;
            }
        }
    }
};

// Update_Error_Derivatives
void TrainingAlgorithmsModel::Update_Error_Derivatives(std::vector < double * > evaldata, int itbgn, int itend) {

    int dimnetin, lyr_nbr;
    int *lyr_cfg, *wgt_cfg;
    double error_, axdf_du_; // Auxiliar variable
    double * _derror_dresp;
    std::vector < std::vector <double *> > dreg_dwgt_cont;

    // Cleaning the last structure
    Clean_dErrordWeight();

    dimnetin = _network->dimnetin(); // Dimension of inlet
    lyr_nbr =  _network->layers();   // Number of layers
    lyr_cfg =  _network->layercfg(); // Array with neuron number in each layer
    wgt_cfg = _network->wgtcfg();    // Array with weight number in the neuron of each layer
    for (int dt_=itbgn; dt_ < itend; dt_++) {

        _network->Calculate_Network(evaldata[dt_]);

        // Updating the error value
        error_ = _errorfunction->Error(_network->rsltcont()[lyr_nbr-1], &evaldata[dt_][dimnetin], _network->wghtcont());
        _globalerror += error_;

        // Calculating the derivative of the error related with the last neuron layer
        _derror_dresp = _errorfunction->dNeterror_dResp(_network->rsltcont()[lyr_nbr-1], &evaldata[dt_][dimnetin]);

        // Updating the derivatives of the error for each network layer
        // Starting with the last layer
        // dE/dw = derror_dresp*_actvfdercont*Y[n-1]
        // dE/dw = _derrordu*Y[n-1]

        // For each Neuron in the last layer
        for (int nrn_=0; nrn_ <  lyr_cfg[lyr_nbr-1]; nrn_++) {
            // For each input for the neuron in last layer
            _derrordu[lyr_nbr-1][nrn_] = _derror_dresp[nrn_]*_network->actvfdercont()[lyr_nbr-1][nrn_];

            // _derrordweight = _derrordu*_rsltcont + _derrordweight
            cblas_daxpy(wgt_cfg[lyr_nbr-1], _derrordu[lyr_nbr-1][nrn_], _network->rsltcont()[lyr_nbr-2], 1, _derrordweight[lyr_nbr-1][nrn_], 1);
        }

        // Updating the Hidden layers
        // For each layer, counting backward:
        for (int hl_= lyr_nbr-2; hl_ > 0; hl_--) {
            // For each Neuron in the layer hl_
            for (int hlnrn_ = 0;  hlnrn_ < lyr_cfg[hl_]; hlnrn_++) {

                // Updating the _derror_du container
                // For each neuron in the forward layer
                axdf_du_ = 0.;

                for (int fhlnrn_ = 0; fhlnrn_ < lyr_cfg[hl_+1]; fhlnrn_++) {
                    axdf_du_ += _derrordu[hl_+1][fhlnrn_]*_network->wghtcont()[hl_+1][fhlnrn_][hlnrn_];
                }

                _derrordu[hl_][hlnrn_] = axdf_du_*_network->actvfdercont()[hl_][hlnrn_];

            // _derrordweight = _derrordu*_rsltcont + _derrordweight
            cblas_daxpy(wgt_cfg[hl_], _derrordu[hl_][hlnrn_], _network->rsltcont()[hl_-1], 1, _derrordweight[hl_][hlnrn_], 1);
            }
        }

        // Updating the First layer
        int hl_ = 0;

        // For each Neuron in the first layer
        for (int hlnrn_=0; hlnrn_ < lyr_cfg[0]; hlnrn_++ ) {

            // Updating the _derror_du container
            // For each neuron in the forward layer
            axdf_du_ = 0.;
            for (int fhlnrn_=0; fhlnrn_ < lyr_cfg[1]; fhlnrn_++) {
                axdf_du_ += _derrordu[1][fhlnrn_]*_network->wghtcont()[1][fhlnrn_][hlnrn_];
            }

            _derrordu[0][hlnrn_] = axdf_du_*_network->actvfdercont()[0][hlnrn_];

            // _derrordweight = _derrordu*_rsltcont + _derrordweight
            cblas_daxpy(wgt_cfg[0], _derrordu[0][hlnrn_], _network->iptcont(), 1, _derrordweight[0][hlnrn_], 1);
        }

        // Updating the error derivative related to the weight due to normalization
        _errorfunction->Calculate_dRegularization_dWeight(_network->wghtcont());
        dreg_dwgt_cont = _errorfunction->dreg_dwgt_cont();

        for (int lyr_= 0; lyr_ < lyr_nbr; lyr_++) {
            for (int nrn_ = 0;  nrn_ < lyr_cfg[lyr_]; nrn_++) {
                for (int wgt_=0; wgt_< wgt_cfg[lyr_]; wgt_++) {
                    // For each weight in network
                    _derrordweight[lyr_][nrn_][wgt_] += dreg_dwgt_cont[lyr_][nrn_][wgt_];
                }
            }
        }
    }

    // Normalizing the global error
    _globalerror = _globalerror / double(itend - itbgn);

    // Normalizing the error derivatives
    for (int lyr_=0; lyr_ < lyr_nbr; lyr_++ ) {
        for (int nrn_=0; nrn_<lyr_cfg[lyr_]; nrn_++) {
            // _derrordweight = _derrordweight / len(evaldata)
            cblas_dscal (wgt_cfg[lyr_], 1./double(itend - itbgn), _derrordweight[lyr_][nrn_], 1);
        }
    }
};

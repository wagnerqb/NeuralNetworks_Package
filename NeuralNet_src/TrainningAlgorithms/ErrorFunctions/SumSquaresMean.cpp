/***************************************************************************//**
 * @file     SumSquaresMean.cpp
 * @date     03 May 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup ErrorFunctions
 * @brief    Class responsible for the error calculus between the trainning
 *           data and the network response using the sum of squares mean model.
 ******************************************************************************/

#include "SumSquaresMean.h"

// SumSquaresMean Constructor
SumSquaresMean::SumSquaresMean(int layers, int * wgtcfg, int * layercfg, double lbd) {

    // Assigning internal values
    _layers = layers;  // Number of layers in network;
    _lbd = lbd;        // Regularization parameter weight

    // Calculates the number of network output
    _outnetnbr = layercfg[_layers-1];

    // Initiating internal containers
    _diffvec = new double [_outnetnbr];
    _dNeterr_dresp = new double [_outnetnbr];

    // Creating an internal copy of weight configuration
    _wgtcfg = new int[_layers];
    for (int lyr_=0; lyr_<_layers; lyr_++) {
    _wgtcfg[lyr_] = wgtcfg[lyr_];
    }

    // Creating an internal copy of layer configuration
    _layercfg = new int[_layers];
    for (int lyr_=0; lyr_<_layers; lyr_++) {
    _layercfg[lyr_] = layercfg[lyr_];
    }

    // Calculates the total number of weights
    _totalwgtnbr = 0;
    for (int lyr_=0; lyr_<_layers; lyr_++) {
        _totalwgtnbr += _layercfg[lyr_]*_wgtcfg[lyr_];
    }

    // Initiating the container _dreg_dwgt_cont
    for (int lyr_=0; lyr_<_layers; lyr_++) {
        // Step 1: Initiating the layer vector
        std::vector <double *> nrn_wgt_; // Creating an empty neurons pointer vector;
        _dreg_dwgt_cont.push_back(nrn_wgt_);
        for (int nrn_=0; nrn_<_layercfg[lyr_]; nrn_++) {
            // Step 2: Initiating the weight array for each neuron in each layer
            _dreg_dwgt_cont[lyr_].push_back(new double [_wgtcfg[lyr_]]);
            for (int wgt_=0; wgt_<_wgtcfg[lyr_]; wgt_++) {
                // Step 3: Assign the 0 value for all creted variable
                _dreg_dwgt_cont[lyr_][nrn_][wgt_] = 0.;
            }
        }
    }

};

// ~SumSquaresMean Desctructor
SumSquaresMean::~SumSquaresMean(void) {

    // Destructing the weights container vector
    for (int lyr_=0; lyr_<_layers; lyr_++) {
        for (int nrn_=0; nrn_<_layercfg[lyr_]; nrn_++) {
            delete _dreg_dwgt_cont[lyr_][nrn_];
        }
    }

    delete _diffvec;
    delete _dNeterr_dresp;
    delete _wgtcfg;
    delete _layercfg;
};

// Error
double SumSquaresMean::Error(double *netresponse, double *expresponse, std::vector < std::vector <double *> > netweights) {
    return Network_Error(netresponse, expresponse) + Regularization(netweights);
};

// Network_Error
double SumSquaresMean::Network_Error(double *netresponse, double *expresponse) {
    // The error will be calculated using the L2 norm, or the squares sum
    double euc_norm_=0.;

    // Step 1: Creating a copy of the network output;
    cblas_dcopy(_outnetnbr, netresponse, 1, _diffvec, 1);

    // Step 2: Performing _diffvec = _diffvec - expresponse operation;
    cblas_daxpy(_outnetnbr, -1.0, expresponse, 1, _diffvec, 1);

    // Step 3: Calculating the squares sum;
    euc_norm_ = cblas_ddot(_outnetnbr, _diffvec, 1, _diffvec, 1);

    euc_norm_ = euc_norm_/2.;

    return euc_norm_;
};

// dNeterror_dResp
double * SumSquaresMean::dNeterror_dResp(double *netresponse, double *expresponse) {

    // Step 1: Creating a copy of the network response;
    cblas_dcopy(_outnetnbr, netresponse, 1, _dNeterr_dresp, 1);

    // Step 2: Performing _dNeterr_dresp = _dNeterr_dresp - expresponse operation;
    cblas_daxpy(_outnetnbr, -1.0, expresponse, 1, _dNeterr_dresp, 1);

    return _dNeterr_dresp;
};

// Regularization
double SumSquaresMean::Regularization(std::vector < std::vector <double *> > net_wghtcont) {

    double reg_sqrsum_;  // Regularization squares sum

    if (_lbd == 0.) {
        // Case without regularization
        return 0.;

    } else {

        for (int lyr_=0; lyr_<_layers; lyr_++) {
            for (int nrn_=0; nrn_<_layercfg[lyr_]; nrn_++) {
                // Performing dot product between xin and weights
                reg_sqrsum_ = reg_sqrsum_ + cblas_ddot(_wgtcfg[lyr_], net_wghtcont[lyr_][nrn_], 1, net_wghtcont[lyr_][nrn_], 1);
            }
        }
        return _lbd*reg_sqrsum_/(2.*double(_totalwgtnbr));
    }
};

// dRegularization_dWeight
void SumSquaresMean::Calculate_dRegularization_dWeight(std::vector < std::vector <double *> > net_wghtcont) {

    if (_lbd == 0.) {
        // Case without regularization
        return;

    } else {

        for (int lyr_=0; lyr_<_layers; lyr_++) {
            for (int nrn_=0; nrn_<_layercfg[lyr_]; nrn_++) {
                // Performing _dreg_dwgt_cont = (_lbd/double(_totalwgtnbr))*_rsltcont + _dreg_dwgt_cont
                cblas_daxpy(_wgtcfg[lyr_], (_lbd/double(_totalwgtnbr)), net_wghtcont[lyr_][nrn_], 1, _dreg_dwgt_cont[lyr_][nrn_], 1);
            }
        }

        return;
    }

};

/***************************************************************************//**
 * @file     GeneralizedError.cpp
 * @date     04 Jun 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup ErrorFunctions
 * @brief    Class responsible for the error calculus between the trainning
 *           data and the network response using the generalized error model.
 ******************************************************************************/

#include "GeneralizedError.h"



// GeneralizedError Constructor
GeneralizedError::GeneralizedError(int layers, int * wgtcfg, int * layercfg, double lbd, double beta)
    : SumSquaresMean(layers, wgtcfg, layercfg, lbd) {

        _beta = beta;

    // Initiating internal containers
    _crss_entrp = new double [_outnetnbr];
    _dcrsetp_dresp = new double [_outnetnbr];
};

// ~GeneralizedError Desctructor
GeneralizedError::~GeneralizedError(void) {

    delete _crss_entrp;
    delete _dcrsetp_dresp;
};


// Network_Error
double GeneralizedError::Network_Error(double *netresponse, double *expresponse) {

    double lstq_pt_, crsetp_=0.;

    // Step 1: Calculating the Least Squares Part of Error
    lstq_pt_ = SumSquaresMean::Network_Error(netresponse, expresponse);
    lstq_pt_ *=  _beta;

    // Step 2: Calculating the Cross Entropy Part of Error
    for (int _ot=0; _ot < _outnetnbr; _ot++) {
        _crss_entrp[_ot] = expresponse[_ot]*log(netresponse[_ot]);
        _crss_entrp[_ot] += (1. - expresponse[_ot])*log(1. - netresponse[_ot]);
        crsetp_ += _crss_entrp[_ot];
    }
    crsetp_ *= -1.*(1. - _beta);

    return lstq_pt_ + crsetp_;
};

// dNeterror_dResp
double * GeneralizedError::dNeterror_dResp(double *netresponse, double *expresponse) {

    double *dNet_dresp;

    // Step 1: Least Squares Part of derivative
    dNet_dresp = SumSquaresMean::dNeterror_dResp(netresponse, expresponse);

    // Step 2: Cross Entropy Part of derivative
    for (int _ot=0; _ot < _outnetnbr; _ot++) {
        _dcrsetp_dresp[_ot] = expresponse[_ot]/netresponse[_ot];
        _dcrsetp_dresp[_ot] += - (1. - expresponse[_ot])/(1. - netresponse[_ot]);
    }

    // Step 3: Summing both derivatives
    for (int _ot=0; _ot < _outnetnbr; _ot++) {
        _dcrsetp_dresp[_ot] = _beta*dNet_dresp[_ot] - (1. - _beta)*_dcrsetp_dresp[_ot];
    }

    return _dcrsetp_dresp;

};

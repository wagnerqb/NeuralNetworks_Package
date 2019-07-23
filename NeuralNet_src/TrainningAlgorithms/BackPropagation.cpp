/***************************************************************************//**
 * @file     BackPropagation.cpp
 * @date     05 Jun 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Networks
 * @brief    This class is responsable for create a training classes based
 *           on the back propagation algorithm.
 ******************************************************************************/

#include "BackPropagation.h"

// Training
void BackPropagation::Training(std::vector < double * > trainingdata, double eta, int max_epoch,
                               double tol, double derror_tol, double batch_ratio, int reportfreq, bool reporttime, bool early_stop) {

    double * w0, * dE_dw; // Auxiliar variables
    double error_ = 1.0;
    int epoch_ = 0;
    clock_t ini_time, final_time;

    ini_time = clock();

    while ((abs(error_) > tol) and (epoch_ <= max_epoch)) {

        error_ = 0.;

        for (int dt_=0; dt_ < trainingdata.size(); dt_++) {
            // Cleaning the auxiliar vector with training data

            // For each Neuron Layer counting backward:
            for (int lyr_=_network->layers()-1; lyr_ >= 0; lyr_--) {

                // Updating the error and derivatives
                Update_Error_Derivatives(trainingdata, dt_, dt_+1);

                // Adjusting the weights for layer lyr_
                for ( int nrn_=0; nrn_ < _network->layercfg()[lyr_]; nrn_++ ) {

                    w0 = _network->wghtcont()[lyr_][nrn_];
                    dE_dw = _derrordweight[lyr_][nrn_];

                    // new_weights = w0 - eta*dE_dw
                    // w0 = eta*dE_dw + w0
                    cblas_daxpy(_network->wgtcfg()[lyr_], -eta, dE_dw, 1, w0, 1);

                    _network->UpdateNeuronWeight(lyr_, nrn_, w0);

                }
            }

            error_ += abs(_globalerror);
        }

        // print "Weights: ", self._network._wghtcont

        if (epoch_ % reportfreq == 0) {
            std::cout <<  "Actual Epoch: " <<  epoch_ << " Error: " <<  abs(error_) << std::endl;
        }
        epoch_ += 1;

    }

    final_time = clock();
    double elap_time;
    elap_time = double(final_time - ini_time) / CLOCKS_PER_SEC;

    if (reporttime) {
        std::cout << "Elapsed Time (s): " <<  std::setprecision(5) << elap_time << std::endl;
    }

};

/***************************************************************************//**
 * @file     DescendingGradient.h
 * @date     10 Jun 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup Networks
 * @brief    This class is responsable for create a training class based
 *           on the descending gradient algorithm.
 ******************************************************************************/

#include "DescendingGradient.h"

// Training
void DescendingGradient::Training(std::vector < double * > trainingdata, double eta, int max_epoch,
                      double tol, double derror_tol, double batch_ratio, int reportfreq, bool reporttime, bool early_stop) {

    double error_ = 1.0;
    double deltaerror_ = 1.0;
    double last_validation_error = 0.0;
    bool pos_derror = false;  // Flag if it has appeared a possitive delta error
    int epoch_ = 0;

    clock_t ini_time, final_time;
    unsigned clock_seed = clock();  // variable used to obtain a time-based seed:
    std::default_random_engine rdn_eng(clock_seed);  // Class used for generating random numbers

    std::vector < double * > validation_data_;
    double * w0, * dE_dw; // Auxiliar variables
    double lsterror_, new_validation_error, dValErr, elap_time;  // Auxiliar variables
    int btch_step, final_btch_idx;  // Auxiliar variables

    ini_time = clock();

    if (early_stop) {
        // 10% of data will be saved as validation data
        double vald_size = 0.10;

        // Shuffling data
        clock_seed = clock();  // obtain a time-based seed:
        rdn_eng.seed(clock_seed);
        std::shuffle(trainingdata.begin(), trainingdata.end(), rdn_eng);

        for (int vldt_ = 0; vldt_ <= int(vald_size*trainingdata.size()); vldt_++) {
            validation_data_.push_back(trainingdata[trainingdata.size()-1]);
            trainingdata.pop_back();
        }

        // Updating the validation error
        Update_Error_Derivatives(validation_data_, 0, validation_data_.size());
        last_validation_error = abs(_globalerror);

    }

    while ( (abs(error_) > tol) and (epoch_ <= max_epoch) and (abs(deltaerror_) > derror_tol) ) {

        lsterror_ = error_;
        error_ = 0.;

        // Shuffling data
        clock_seed = clock();  // obtain a time-based seed:
        rdn_eng.seed(clock_seed);
        std::shuffle(trainingdata.begin(), trainingdata.end(), rdn_eng);

        btch_step = int(batch_ratio*trainingdata.size());
        if (btch_step < 1) {
            btch_step = 1;
        }

        for (int ini_btch_idx = 0; ini_btch_idx < trainingdata.size()-1; ini_btch_idx += btch_step) {

            final_btch_idx = ini_btch_idx + btch_step;
            if (final_btch_idx > trainingdata.size()) {
                final_btch_idx = trainingdata.size();
            }

            // Updating the error and derivatives
            Update_Error_Derivatives(trainingdata, ini_btch_idx, final_btch_idx);

            // For each Neuron Layer counting backward:
            for (int lyr_=_network->layers()-1; lyr_ >= 0; lyr_--) {

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

            // Updating the iteration error
            error_ += abs(_globalerror)*batch_ratio;
        }

        deltaerror_ = abs(error_) - lsterror_;
        if (deltaerror_ > 0.0) {
            pos_derror = true;
        }

        if (early_stop) {
            if ( (epoch_ % int(0.05*max_epoch) == 0) and (pos_derror) ) {

                // Updating the validation error
                Update_Error_Derivatives(validation_data_, 0, validation_data_.size());
                new_validation_error = abs(_globalerror);

                dValErr = new_validation_error - last_validation_error;
                last_validation_error = new_validation_error;
                if (epoch_ % reportfreq == 0) {
                    std::cout << "Epoch: " <<  epoch_ << ", Error: " << std::setprecision(5) << abs(error_);
                    std::cout << ", Valid.Err: " << std::setprecision(5) << abs(new_validation_error);
                    std::cout << ", DValidErr: " << std::setprecision(5) << dValErr << std::endl;
                }

                if ( (epoch_ > int(0.3*max_epoch)) and (dValErr > tol) ) {
                    std::cout << "Early stop criteria acchieved" << std::endl;

                    std::cout << "Epoch: " <<  epoch_ << ", Error: " << std::setprecision(5) << abs(error_);
                    std::cout << ", Valid.Err: " << std::setprecision(5) << abs(new_validation_error) << dValErr << std::endl;

                    if (reporttime) {
                        final_time = clock();
                        elap_time = double(final_time - ini_time) / CLOCKS_PER_SEC;
                        std::cout << "Elapsed Time (s): " <<  std::setprecision(5) << elap_time << std::endl;
                    }

                    // Destroying validation data
                    for (int vd_=0; vd_< validation_data_.size(); vd_++) {
                        trainingdata.push_back(validation_data_[validation_data_.size()-1]);
                        validation_data_.pop_back();
                    }
                    return;
                }

            } else {
                if (epoch_ % reportfreq == 0) {
                    std::cout << "Epoch: " <<  epoch_ << ", Error: " << std::setprecision(5) << abs(error_);
                    std::cout << ", Delta Error: " << std::setprecision(5) << deltaerror_ << std::endl;
                }
            }

        } else {
            if (epoch_ % reportfreq == 0) {
                std::cout << "Epoch: " <<  epoch_ << ", Error: " << std::setprecision(5) << abs(error_);
                std::cout << ", Delta Error: " << std::setprecision(5) << deltaerror_ << std::endl;
            }
        }

        epoch_ += 1;

    }

    if (reporttime) {
        final_time = clock();
        elap_time = double(final_time - ini_time) / CLOCKS_PER_SEC;
        std::cout << "Elapsed Time (s): " <<  std::setprecision(5) << elap_time << std::endl;
    }

    if (early_stop) {
        // Destroying validation data
        for (int vd_=0; vd_< validation_data_.size(); vd_++) {
            trainingdata.push_back(validation_data_[validation_data_.size()-1]);
            validation_data_.pop_back();
        }
    }

    return;
};

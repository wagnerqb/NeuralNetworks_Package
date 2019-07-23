#include <iostream>
#include "UnitTestPackage.h"
#include "Perceptron.h"

/***************************************************************************//**
 * @class Test_Perceptron
 * @brief This class is the Unit Test for Perceptron Class
 * @ingroup Test_NeuralNetPackage
 ******************************************************************************/
class Test_Perceptron : public UnitTestPackage {

public:

    /*************************************************************************//**
     * Constructs a Test_Perceptron object
     ****************************************************************************/
    // Default constructor
    Test_Perceptron() {
        UnitTestPackage();

        // Creating the Neuron:
        // double weights[3];
        // weights[0] = 0.15;
        // weights[1] = -0.45;
        // weights[2] = 0.3;
        double * weights = NULL;
        _nrn = new Perceptron(3, weights, true);

    };

    Perceptron *_nrn;

    // Default destructor
    virtual ~Test_Perceptron() {
        delete _nrn;
    };

    // Run_Tests
    void Run_Tests(void) {

        test_update_weights();

        Generate_TestLog("Perceptron");
    }

    // Tests
    void test_update_weights(void) {
        //Tests the update_weights calculation of the Perceptron method

        double new_wgts[3];
        new_wgts[0] = -1.2;
        new_wgts[1] = 0.19;
        new_wgts[2] = 0.1;

        _nrn->update_weights(new_wgts);

        new_wgts[0] = -1.4;
        new_wgts[1] = 0.3;
        new_wgts[2] = 0.18;

        AssertDoubleEqual(_nrn->weights()[0], -1.2, "Error in Update Weights");
        AssertDoubleEqual(_nrn->weights()[1], 0.19, "Error in Update Weights");
        AssertDoubleEqual(_nrn->weights()[2], 0.10, "Error in Update Weights");

    }

};

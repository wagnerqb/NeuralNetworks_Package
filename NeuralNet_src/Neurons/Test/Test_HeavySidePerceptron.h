#include <iostream>
#include "UnitTestPackage.h"
#include "Perceptron.h"
#include "HeavySidePerceptron.h"

/***************************************************************************//**
 * @class Test_HeavySidePerceptron
 * @brief This class is the Unit Test for HeavySidePerceptron Class
 * @ingroup Test_NeuralNetPackage
 ******************************************************************************/
class Test_HeavySidePerceptron : public UnitTestPackage {

public:

    /*************************************************************************//**
     * Constructs a Test_HeavySidePerceptron object
     ****************************************************************************/
    // Default constructor
    Test_HeavySidePerceptron() {
        UnitTestPackage();

        // Creating the Neuron:
        double weights[3];
        weights[0] = 0.15;
        weights[1] = -0.45;
        weights[2] = 0.3;
        _nrn = new HeavySidePerceptron(3, weights, true);

        // Adding the Neuron Input for testing
        _nrninput = new double(3);
        _nrninput[0] = 0.2;
        _nrninput[1] = -0.1;
        _nrninput[2] = 0.0;

    };

    Perceptron *_nrn;
    double *_nrninput;

    // Default destructor
    virtual ~Test_HeavySidePerceptron() {
        delete _nrn;
        delete _nrninput;
    };

    // Run_Tests
    void Run_Tests(void) {

        test_ActvF();
        test_dActvF_du();
        test_Neural_Clone();

        Generate_TestLog("HeavySidePerceptron");
    }

    // Tests
    void test_ActvF(void) {
        //Tests the ActvF calculation of the HeavySidePerceptron class
        AssertAlmostEqual(_nrn->ActvF(0.5), 1.0, 1.0e-8, "Error in ActvF");
        AssertAlmostEqual(_nrn->ActvF(-0.5), 0.0, 1.0e-8, "Error in ActvF");
    }

    void test_dActvF_du(void) {
        //Tests the dActvF_du calculation of the HeavySidePerceptron class
        double eps_ = 1.e-6;
        double dActvF_du_;

        dActvF_du_ = (_nrn->ActvF(0.5 + eps_) - _nrn->ActvF(0.5))/eps_;
        AssertAlmostEqual(_nrn->dActvF_du(0.5), dActvF_du_, 1.0e-8, "Error in dActvF_du");
    }

    void test_Neural_Clone(void) {
        //Tests the clone function of the HeavySidePerceptron class
        Perceptron *percnrn;

        percnrn = _nrn->Clone();
        AssertAlmostEqual(percnrn->ActvF(0.5), 1.0, 1.0e-8, "Error in ActvF");
        AssertAlmostEqual(percnrn->ActvF(-0.5), 0.0, 1.0e-8, "Error in ActvF");

    }

};

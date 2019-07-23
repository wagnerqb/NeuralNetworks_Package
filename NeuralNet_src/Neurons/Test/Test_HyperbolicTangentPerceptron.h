#include <iostream>
#include "UnitTestPackage.h"
#include "HyperbolicTangentPerceptron.h"

/***************************************************************************//**
 * @class Test_HyperbolicTangentPerceptron
 * @brief This class is the Unit Test for HyperbolicTangentPerceptron Class
 * @ingroup Test_NeuralNetPackage
 ******************************************************************************/
class Test_HyperbolicTangentPerceptron : public UnitTestPackage {

public:

    /*************************************************************************//**
     * Constructs a Test_HyperbolicTangentPerceptron object
     ****************************************************************************/
    // Default constructor
    Test_HyperbolicTangentPerceptron() {
        UnitTestPackage();

        // Creating the Neuron:
        double weights[3];
        weights[0] = 0.15;
        weights[1] = -0.45;
        weights[2] = 0.3;
        _nrn = new HyperbolicTangentPerceptron(3, weights, true, 1.);

        // Adding the Neuron Input for testing
        _nrninput = new double(3);
        _nrninput[0] = 0.2;
        _nrninput[1] = -0.1;
        _nrninput[2] = 0.0;

    };

    Perceptron *_nrn;
    double *_nrninput;

    // Default destructor
    virtual ~Test_HyperbolicTangentPerceptron() {
        delete _nrn;
        delete _nrninput;
    };

    // Run_Tests
    void Run_Tests(void) {

        test_ActvF();
        test_dActvF_du();
        test_Neural_Clone();

        Generate_TestLog("HyperbolicTangentPerceptron");
    }

    // Tests
    void test_ActvF(void) {
        //Tests the ActvF calculation of the HyperbolicTangentPerceptron class
        AssertAlmostEqual(_nrn->ActvF(0.5), 0.24491866240370913, 1.0e-8, "Error in ActvF");
    }

    void test_dActvF_du(void) {
        //Tests the dActvF_du calculation of the HyperbolicTangentPerceptron class
        double eps_ = 1.e-6;
        double dActvF_du_;

        dActvF_du_ = (_nrn->ActvF(0.5 + eps_) - _nrn->ActvF(0.5))/eps_;
        AssertAlmostEqual(_nrn->dActvF_du(0.5), dActvF_du_, 1.0e-6, "Error in dActvF_du");
    }

    void test_Neural_Clone(void) {
        //Tests the clone function of the HeavySidePerceptron class
        Perceptron *percnrn;

        percnrn = _nrn->Clone();
        AssertAlmostEqual(percnrn->ActvF(0.5), 0.24491866240370913, 1.0e-8, "Error in ActvF");

    }

};

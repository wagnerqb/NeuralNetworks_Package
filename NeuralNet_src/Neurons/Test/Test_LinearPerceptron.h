#include <iostream>
#include "UnitTestPackage.h"
#include "LinearPerceptron.h"

/***************************************************************************//**
 * @class Test_LinearPerceptron
 * @brief This class is the Unit Test for LinearPerceptron Class
 * @ingroup Test_NeuralNetPackage
 ******************************************************************************/
class Test_LinearPerceptron : public UnitTestPackage {

public:

    /*************************************************************************//**
     * Constructs a Test_LinearPerceptron object
     ****************************************************************************/
    // Default constructor
    Test_LinearPerceptron() {
        UnitTestPackage();

        // Creating the Neuron:
        double weights[3];
        weights[0] = 0.15;
        weights[1] = -0.45;
        weights[2] = 0.3;
        _nrn = new LinearPerceptron(3, weights, true);

        // Adding the Neuron Input for testing
        _nrninput = new double(3);
        _nrninput[0] = 0.2;
        _nrninput[1] = -0.1;
        _nrninput[2] = 0.0;

    };

    Perceptron *_nrn;
    double *_nrninput;

    // Default destructor
    virtual ~Test_LinearPerceptron() {
        delete _nrn;
        delete _nrninput;
    };

    // Run_Tests
    void Run_Tests(void) {

        test_ActvF();
        test_dActvF_du();
        test_response();
        test_dresp_du();
        test_Neural_Clone();

        Generate_TestLog("LinearPerceptron");
    }

    // Tests
    void test_ActvF(void) {
        //Tests the ActvF calculation of the LinearPerceptron class
        AssertAlmostEqual(_nrn->ActvF(0.5), 0.5, 1.0e-8, "Error in ActvF");

    }

    void test_dActvF_du(void) {
        //Tests the dActvF_du calculation of the LinearPerceptron class
        AssertAlmostEqual(_nrn->dActvF_du(0.5), 1.0, 1.0e-8, "Error in dActvF_du");

    }

    void test_response(void) {
        //Tests the response calculation of the LinearPerceptron class
        AssertAlmostEqual(_nrn->response(_nrninput), 0.075, 1.0e-8, "Error in response");

    }

    void test_dresp_du(void) {
        //Tests the dresp_du calculation of the LinearPerceptron class
        AssertAlmostEqual(_nrn->dresp_du(_nrninput), 1.0, 1.0e-8, "Error in dresp_du");

    }

    void test_Neural_Clone(void) {
        //Tests the clone function of the HeavySidePerceptron class
        Perceptron *percnrn;

        percnrn = _nrn->Clone();
        AssertAlmostEqual(percnrn->ActvF(0.5), 0.5, 1.0e-8, "Error in ActvF");
    }


};

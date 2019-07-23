#include <iostream>
#include "UnitTestPackage.h"
#include "GeneralizedError.h"

/***************************************************************************//**
 * @class Test_GeneralizedError
 * @brief This class is the Unit Test for GeneralizedError Class
 * @ingroup Test_NeuralNetPackage
 ******************************************************************************/
class Test_GeneralizedError : public UnitTestPackage {

public:

    /*************************************************************************//**
     * Constructs a Test_GeneralizedError object
     ****************************************************************************/
    // Default constructor
    Test_GeneralizedError() {
        UnitTestPackage();
    };

    // Default destructor
    virtual ~Test_GeneralizedError() {
    };

    // Run_Tests
    void Run_Tests(void) {

        test_error_1output_lbd0();
        test_error_1output();
        test_error_3outputs();

        Generate_TestLog("GeneralizedError");
    }

    void test_error_1output_lbd0(void) {
        //Tests the error calculation of the GeneralizedError method
        double *wgt;      // weights for each neuron
        double *ipt;      // Input for the network
        Perceptron *nrn;  // inserted neuron
        GeneralizedError *errorfunction; //Error Function variable

        // Creating an empty network with 2 layers and 4 input dimensions
        bool biasc[] {false, true};

        MLP_Network MLP_Network_(2, 4, biasc);

        // Settup of the first network layer
            // First Neuron:
            wgt = new double[4] {0.1, -0.5, 0.3, 0.2};
            nrn = new SigmoidalPerceptron(4, wgt, false, 1.0, 0.8);

            // Adding the Neuron the network
            MLP_Network_.Insert_Neuron(nrn, 0);

            delete nrn;
            delete wgt;

            // Second Neuron:
            wgt = new double[4] {-0.1, -0.2, 0.4, -0.1};
            nrn = new SigmoidalPerceptron(4, wgt, false, 1.0, 0.7);

            // Adding the Neuron the network
            MLP_Network_.Insert_Neuron(nrn, 0);

            delete wgt;
            delete nrn;

        // Setup the second network layer
            // First Neuron:
            wgt = new double[3] {0.5, -0.7, 0.8};
            nrn = new SigmoidalPerceptron(2, wgt, true, 1.0, 0.9);

            // Adding the Neuron the network
            MLP_Network_.Insert_Neuron(nrn, 1);

            delete wgt;
            delete nrn;

        // Settup error function
        errorfunction = new GeneralizedError(MLP_Network_.layers(), MLP_Network_.wgtcfg(), MLP_Network_.layercfg());

        // Network input vector
        ipt = new double [4] {0.9694,  0.6909,  0.4334,  3.4965};
        double net_response[] {0.16};
        AssertAlmostEqual(errorfunction->Error(MLP_Network_.Response(ipt), net_response, MLP_Network_.wghtcont()),
                          0.014247347358121451, 1.0e-8, "Error in Generalized Error calculation");
        AssertAlmostEqual(errorfunction->dNeterror_dResp(MLP_Network_.Response(ipt), net_response)[0],
                          0.16880371653563467, 1.0e-8, "Error in Sum Squares Mean derivative calculation");

        delete ipt;
        delete errorfunction;

    }

    void test_error_1output(void) {
        //Tests the error calculation of the GeneralizedError method with regularization
        double *wgt;      // weights for each neuron
        double *ipt;      // Input for the network
        Perceptron *nrn;  // inserted neuron
        GeneralizedError *errorfunction; //Error Function variable

        // Creating an empty network with 2 layers and 4 input dimensions
        bool biasc[] {false, true};

        MLP_Network MLP_Network_(2, 4, biasc);

        // Settup of the first network layer
            // First Neuron:
            wgt = new double[4] {0.1, -0.5, 0.3, 0.2};
            nrn = new SigmoidalPerceptron(4, wgt, false, 1.0, 0.8);

            // Adding the Neuron the network
            MLP_Network_.Insert_Neuron(nrn, 0);

            delete nrn;
            delete wgt;

            // Second Neuron:
            wgt = new double[4] {-0.1, -0.2, 0.4, -0.1};
            nrn = new SigmoidalPerceptron(4, wgt, false, 1.0, 0.7);

            // Adding the Neuron the network
            MLP_Network_.Insert_Neuron(nrn, 0);

            delete wgt;
            delete nrn;

        // Setup the second network layer
            // First Neuron:
            wgt = new double[3] {0.5, -0.7, 0.8};
            nrn = new SigmoidalPerceptron(2, wgt, true, 1.0, 0.9);

            // Adding the Neuron the network
            MLP_Network_.Insert_Neuron(nrn, 1);

            delete wgt;
            delete nrn;

        // Settup error function
        errorfunction = new GeneralizedError(MLP_Network_.layers(), MLP_Network_.wgtcfg(), MLP_Network_.layercfg(), 0.2, 0.8);

        // Network input vector
        ipt = new double [4] {0.9694,  0.6909,  0.4334,  3.4965};
        double net_response[] {0.16};
        MLP_Network_.Update_NeuralWeights_Container();
        AssertAlmostEqual(errorfunction->Error(MLP_Network_.Response(ipt), net_response, MLP_Network_.wghtcont()),
                          0.13206274002563287, 1.0e-8, "Error in Generalized Error calculation");
        AssertAlmostEqual(errorfunction->dNeterror_dResp(MLP_Network_.Response(ipt), net_response)[0],
                          0.2880198320474462, 1.0e-8, "Error in Sum Squares Mean derivative calculation");

        delete ipt;
        delete errorfunction;

    }

    void test_error_3outputs(void) {
        //Tests the error calculation of the GeneralizedError method with
        // a neural network with three outputs

        double *wgt;      // weights for each neuron
        double *ipt;      // Input for the network
        Perceptron *nrn;  // inserted neuron
        GeneralizedError *errorfunction; //Error Function variable

        // Creating an empty network with 2 layers and 2 input dimensions
        bool biasc[] {true, true};

        MLP_Network MLP_Network_(2, 2, biasc);

        // Settup of the first network layer
            // First Neuron:
            wgt = new double[3] {0.1, 0.2, 0.9};
            nrn = new SigmoidalPerceptron(2, wgt, true, 1.0, 0.8);

            // Adding the Neuron the network
            MLP_Network_.Insert_Neuron(nrn, 0);

            delete nrn;
            delete wgt;

            // Second Neuron:
            wgt = new double[3] {0.4, -0.1, -0.2};
            nrn = new SigmoidalPerceptron(2, wgt, true, 1.0, 0.7);

            // Adding the Neuron the network
            MLP_Network_.Insert_Neuron(nrn, 0);

            delete wgt;
            delete nrn;

        // Setup the last network layer
            // First Neuron:
            wgt = new double[3] {0.1, -0.9, 0.8};
            nrn = new SigmoidalPerceptron(2, wgt, true, 1.0, 0.7);

            // Adding the Neuron the network
            MLP_Network_.Insert_Neuron(nrn, 1);

            delete wgt;
            delete nrn;

            // Second Neuron:
            wgt = new double[3] {-0.2, 0.9, 0.6};
            nrn = new SigmoidalPerceptron(2, wgt, true, 1.0, 0.7);

            // Adding the Neuron the network
            MLP_Network_.Insert_Neuron(nrn, 1);

            delete wgt;
            delete nrn;

            // Third Neuron:
            wgt = new double[3] {0.1, 0.1, -0.5};
            nrn = new SigmoidalPerceptron(2, wgt, true, 1.0, 0.7);

            // Adding the Neuron the network
            MLP_Network_.Insert_Neuron(nrn, 1);

            delete wgt;
            delete nrn;

        // Settup error function
        errorfunction = new GeneralizedError(MLP_Network_.layers(), MLP_Network_.wgtcfg(), MLP_Network_.layercfg(), 0.23, 0.78);

        // Network input vector
        ipt = new double [3] {0.9694,  0.6909};
        double net_response[] {0.16, -0.25, 0.18};

        MLP_Network_.Update_NeuralWeights_Container();
        AssertAlmostEqual(errorfunction->Error(MLP_Network_.Response(ipt), net_response, MLP_Network_.wghtcont()),
                          0.7412289521155222, 1.0e-8, "Error in Generalized Error calculation");
        AssertAlmostEqual(errorfunction->dNeterror_dResp(MLP_Network_.Response(ipt), net_response)[0],
                          0.23699434794942825, 1.0e-8, "Error in Sum Squares Mean derivative calculation");
        AssertAlmostEqual(errorfunction->dNeterror_dResp(MLP_Network_.Response(ipt), net_response)[1],
                          1.205036508777567, 1.0e-8, "Error in Sum Squares Mean derivative calculation");
        AssertAlmostEqual(errorfunction->dNeterror_dResp(MLP_Network_.Response(ipt), net_response)[2],
                          0.7182914259398843, 1.0e-8, "Error in Sum Squares Mean derivative calculation");

        delete ipt;
        delete errorfunction;

    }

};

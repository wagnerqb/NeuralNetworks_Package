#include <iostream>
#include "UnitTestPackage.h"
#include "SumSquaresMean.h"

/***************************************************************************//**
 * @class Test_SumSquaresMean
 * @brief This class is the Unit Test for SumSquaresMean Class
 * @ingroup Test_NeuralNetPackage
 ******************************************************************************/
class Test_SumSquaresMean : public UnitTestPackage {

public:

    /*************************************************************************//**
     * Constructs a Test_SumSquaresMean object
     ****************************************************************************/
    // Default constructor
    Test_SumSquaresMean() {
        UnitTestPackage();
    };

    // Default destructor
    virtual ~Test_SumSquaresMean() {
    };

    // Run_Tests
    void Run_Tests(void) {

        test_error_1output_lbd0();
        test_error_1output();
        test_error_3outputs();

        Generate_TestLog("SumSquaresMean");
    }

    void test_error_1output_lbd0(void) {
        //Tests the error calculation of the SumSquaresMean method
        double *wgt;      // weights for each neuron
        double *ipt;      // Input for the network
        Perceptron *nrn;  // inserted neuron
        SumSquaresMean *errorfunction; //Error Function variable

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

            nrn = new RectifiedLinearPerceptron(4, wgt, false);

            // Adding the Neuron the network
            MLP_Network_.Insert_Neuron(nrn, 0);

            delete wgt;
            delete nrn;

        // Setup the second network layer
            // First Neuron:
            wgt = new double[3] {0.5, -0.7, 0.8};

            nrn = new LinearPerceptron(2, wgt, true);

            // Adding the Neuron the network
            MLP_Network_.Insert_Neuron(nrn, 1);

            delete wgt;
            delete nrn;

        // Settup error function
        errorfunction = new SumSquaresMean(MLP_Network_.layers(), MLP_Network_.wgtcfg(), MLP_Network_.layercfg());

        // Network input vector
        ipt = new double [4] {0.9694,  0.6909,  0.4334,  3.4965};
        double net_response[] {0.16};
        AssertAlmostEqual(errorfunction->Error(MLP_Network_.Response(ipt), net_response, MLP_Network_.wghtcont()),
                          0.21316655257628575, 1.0e-8, "Error in Sum Squares Mean calculation");
        AssertAlmostEqual(errorfunction->dNeterror_dResp(MLP_Network_.Response(ipt), net_response)[0],
                          -0.6529418849733654, 1.0e-8, "Error in Sum Squares Mean derivative calculation");

        delete ipt;
        delete errorfunction;
    }

    void test_error_1output(void) {
        //Tests the error calculation of the SumSquaresMean method with regularization
        double *wgt;      // weights for each neuron
        double *ipt;      // Input for the network
        Perceptron *nrn;  // inserted neuron
        SumSquaresMean *errorfunction; //Error Function variable

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

            nrn = new RectifiedLinearPerceptron(4, wgt, false);

            // Adding the Neuron the network
            MLP_Network_.Insert_Neuron(nrn, 0);

            delete wgt;
            delete nrn;

        // Setup the second network layer
            // First Neuron:
            wgt = new double[3] {0.5, -0.7, 0.8};

            nrn = new LinearPerceptron(2, wgt, true);

            // Adding the Neuron the network
            MLP_Network_.Insert_Neuron(nrn, 1);

            delete wgt;
            delete nrn;

        // Settup error function
        errorfunction = new SumSquaresMean(MLP_Network_.layers(), MLP_Network_.wgtcfg(), MLP_Network_.layercfg(), 0.5);

        // Network input vector
        ipt = new double [4] {0.9694,  0.6909,  0.4334,  3.4965};
        double net_response[] {0.16};
        MLP_Network_.Update_NeuralWeights_Container();
        AssertAlmostEqual(errorfunction->Error(MLP_Network_.Response(ipt), net_response, MLP_Network_.wghtcont()),
                          0.2583938253035585, 1.0e-8, "Error in Sum Squares Mean calculation");
        AssertAlmostEqual(errorfunction->dNeterror_dResp(MLP_Network_.Response(ipt), net_response)[0],
                          -0.6529418849733654, 1.0e-8, "Error in Sum Squares Mean derivative calculation");

        delete ipt;
        delete errorfunction;

    }

    void test_error_3outputs(void) {
        //Tests the error calculation of the SumSquaresMean method with
        // a neural network with three outputs

        double *wgt;      // weights for each neuron
        double *ipt;      // Input for the network
        Perceptron *nrn;  // inserted neuron
        SumSquaresMean *errorfunction; //Error Function variable

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

            nrn = new RectifiedLinearPerceptron(2, wgt, true);

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
        errorfunction = new SumSquaresMean(MLP_Network_.layers(), MLP_Network_.wgtcfg(), MLP_Network_.layercfg(), 0.8);

        // Network input vector
        ipt = new double [3] {0.9694,  0.6909};
        double net_response[] {0.16, -0.25, 0.18};

        MLP_Network_.Update_NeuralWeights_Container();
        AssertAlmostEqual(errorfunction->Error(MLP_Network_.Response(ipt), net_response, MLP_Network_.wghtcont()),
                          0.45997697651905267, 1.0e-8, "Error in Sum Squares Mean calculation");
        AssertAlmostEqual(errorfunction->dNeterror_dResp(MLP_Network_.Response(ipt), net_response)[0],
                          0.13716032581442775, 1.0e-8, "Error in Sum Squares Mean derivative calculation");
        AssertAlmostEqual(errorfunction->dNeterror_dResp(MLP_Network_.Response(ipt), net_response)[1],
                          0.7138014155913865, 1.0e-8, "Error in Sum Squares Mean derivative calculation");
        AssertAlmostEqual(errorfunction->dNeterror_dResp(MLP_Network_.Response(ipt), net_response)[2],
                          0.42161815721533075, 1.0e-8, "Error in Sum Squares Mean derivative calculation");

        delete ipt;
        delete errorfunction;

    }

};

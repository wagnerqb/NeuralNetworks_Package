#include <iostream>
#include "UnitTestPackage.h"
#include "MLP_Network.h"
#include "TrainingAlgorithmsModel.h"
#include "BackPropagation.h"

/***************************************************************************//**
 * @class Test_TrainingAlgorithmsModel
 * @brief This class is the Unit Test for TrainingAlgorithmsModel Class
 * @ingroup Test_NeuralNetPackage
 ******************************************************************************/
class Test_TrainingAlgorithmsModel : public UnitTestPackage {

public:

    /*************************************************************************//**
     * Constructs a Test_TrainingAlgorithmsModel object
     ****************************************************************************/
    // Default constructor
    Test_TrainingAlgorithmsModel() {
        UnitTestPackage();
    };

    // Default destructor
    virtual ~Test_TrainingAlgorithmsModel() {

    };

    // Run_Tests
    void Run_Tests(void) {

        test_update_error_derivatives();

        Generate_TestLog("TrainingAlgorithmsModel");
    }

    // Tests
    void test_update_error_derivatives(void) {
        //Tests the TrainingAlgorithmsModel with three neurons un last layer
        double *wgt;      // weights for each neuron
        double *ipt;      // Input for the network
        Perceptron *nrn;  // inserted neuron
        TrainingAlgorithmsModel *TrainingModel;  // Model for Training Algorithm

        // Creating an empty network with layercfg = (2, 3, 4, 2)
        bool biasc[] {false, true, true, true};

        MLP_Network network(4, 4, biasc);

        // Settup of the first network layer
            // First Neuron:
            wgt = new double[4] {0.2, 0.15, -0.1, 0.5};
            nrn = new SigmoidalPerceptron(4, wgt, biasc[0], 1.0, 1.0);
            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 0);

            delete nrn;
            delete wgt;

            // Second Neuron:
            wgt = new double[4] {0.1, -0.05, 0.14, 0.3};
            nrn = new SigmoidalPerceptron(4, wgt, biasc[0], 1.0, 1.0);
            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 0);

            delete wgt;
            delete nrn;

        // Settup of the second network layer
            // First Neuron:
            wgt = new double[3] {0.18, -0.12, -0.1};
            nrn = new SigmoidalPerceptron(2, wgt, biasc[1], 1.0, 1.0);
            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 1);

            delete wgt;
            delete nrn;

            // Second Neuron:
            wgt = new double[3] {-0.18, 0.12, 0.1};
            nrn = new SigmoidalPerceptron(2, wgt, biasc[1], 1.0, 1.0);
            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 1);

            delete wgt;
            delete nrn;

            // Third Neuron:
            wgt = new double[3] {0.01, 0.15, 0.0};
            nrn = new SigmoidalPerceptron(2, wgt, biasc[1], 1.0, 1.0);
            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 1);

            delete wgt;
            delete nrn;

        // Settup of the third network layer
            // First Neuron:
            wgt = new double[4] {0.89, -0.73, 0.41, -0.05};
            nrn = new SigmoidalPerceptron(3, wgt, biasc[2], 1.0, 1.0);
            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 2);

            delete wgt;
            delete nrn;

            // Second Neuron:
            wgt = new double[4] {-0.41, -0.89, -0.89, -0.72};
            nrn = new SigmoidalPerceptron(3, wgt, biasc[2], 1.0, 1.0);
            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 2);

            delete wgt;
            delete nrn;

            // Third Neuron:
            wgt = new double[4] {0.99, -0.35, 0.71, 0.27};
            nrn = new SigmoidalPerceptron(3, wgt, biasc[2], 1.0, 1.0);
            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 2);

            delete wgt;
            delete nrn;

            // Fourth Neuron:
            wgt = new double[4] {0.13, 0.43, 0.51, 0.25};
            nrn = new SigmoidalPerceptron(3, wgt, biasc[2], 1.0, 1.0);
            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 2);

            delete wgt;
            delete nrn;

        // Settup of the last network layer
            // First Neuron:
            wgt = new double[5] {0.58, -0.72, 0.54, -0.03, 0.05};
            nrn = new SigmoidalPerceptron(4, wgt, biasc[3], 1.0, 1.0);
            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 3);

            delete wgt;
            delete nrn;

            // Second Neuron:
            wgt = new double[5] {-0.32, -0.27, 0.31, 0.0, 0.8};
            nrn = new SigmoidalPerceptron(4, wgt, biasc[3], 1.0, 1.0);
            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 3);

            delete wgt;
            delete nrn;

        TrainingModel = new BackPropagation(&network, 1.0, 0.25);

        std::vector < double * > evaldata;
        evaldata.push_back( new double [6] {-0.15, 0.7, -0.2, 0.5, -2., 5.} );

        TrainingModel->Update_Error_Derivatives(evaldata, 0, evaldata.size());

        std::vector < std::vector < double * > > derdweight;
        derdweight = TrainingModel->derrordweight();

        // First Layer
        AssertAlmostEqual(derdweight[0][0][0], 0.0007387750442869913, 1.0e-8, "Error in TrainingModel Response");
        AssertAlmostEqual(derdweight[0][0][3], 0.00432036225547747, 1.0e-8, "Error in TrainingModel Response");
        AssertAlmostEqual(derdweight[0][1][3], 0.0013218584630839145, 1.0e-8, "Error in TrainingModel Response");

        // Second Layer
        AssertAlmostEqual(derdweight[1][1][0], -0.01299535349860034, 1.0e-8, "Error in TrainingModel Response");
        AssertAlmostEqual(derdweight[1][0][2], -0.04332133420872391, 1.0e-8, "Error in TrainingModel Response");
        AssertAlmostEqual(derdweight[1][2][1], 0.015359099357891794, 1.0e-8, "Error in TrainingModel Response");

        // Third Layer
        AssertAlmostEqual(derdweight[2][1][1], -0.026723023955716393, 1.0e-8, "Error in TrainingModel Response");
        AssertAlmostEqual(derdweight[2][0][2], 0.08659589335656459, 1.0e-8, "Error in TrainingModel Response");
        AssertAlmostEqual(derdweight[2][3][1], 0.0003549852649256842, 1.0e-8, "Error in TrainingModel Response");

        // Fourth Layer
        AssertAlmostEqual(derdweight[3][0][0], 0.3799453555383333, 1.0e-8, "Error in TrainingModel Response");
        AssertAlmostEqual(derdweight[3][0][1], 0.2516596161277446, 1.0e-8, "Error in TrainingModel Response");
        AssertAlmostEqual(derdweight[3][1][3], -0.5502427304486327, 1.0e-8, "Error in TrainingModel Response");

        delete evaldata[0];
        delete TrainingModel;
    }

};

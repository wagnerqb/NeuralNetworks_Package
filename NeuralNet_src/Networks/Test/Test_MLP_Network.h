#include <iostream>
#include "UnitTestPackage.h"
#include "MLP_Network.h"

/***************************************************************************//**
 * @class Test_MLP_Network
 * @brief This class is the Unit Test for MLP_Network Class
 * @ingroup Test_NeuralNetPackage
 ******************************************************************************/
class Test_MLP_Network : public UnitTestPackage {

public:

    /*************************************************************************//**
     * Constructs a Test_MLP_Network object
     ****************************************************************************/
    // Default constructor
    Test_MLP_Network() {
        UnitTestPackage();
    };

    // Default destructor
    virtual ~Test_MLP_Network() {

    };

    // Run_Tests
    void Run_Tests(void) {

        test_response();
        test_multiple_responses();
        test_sigmoidal_network();
        test_update_neuralweight_container();
        test_sigmoidal_network_four_layers();
        test_network_three_outputs();

        Generate_TestLog("MLP_Network");
    }

    // Tests
    void test_response(void) {
        //Tests the MLP_Network response method
        double *wgt;      // weights for each neuron
        double *ipt;      // Input for the network
        Perceptron *nrn;  // inserted neuron

        // Creating an empty network with 2 layers and 4 input dimensions
        bool biasc[] {false, true};

        MLP_Network network(2, 4, biasc);

        // Settup of the first network layer
            // First Neuron:
            wgt = new double[4] {0.1, -0.5, 0.3, 0.2};

            nrn = new LinearPerceptron(4, wgt, false);
            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 0);
            delete nrn;
            delete wgt;

            // Second Neuron:
            wgt = new double[4] {-0.1, -0.2, 0.4, -0.1};

            nrn = new LinearPerceptron(4, wgt, false);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 0);

            delete wgt;
            delete nrn;

        // Setup the second network layer
            // First Neuron:
            wgt = new double[3] {0.5, -0.7, 0.8};

            nrn = new LinearPerceptron(2, wgt, true);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 1);

            delete wgt;
            delete nrn;

        // Creating the input vector
        ipt = new double [4] {0.9694,  0.6909,  0.4334,  3.4965};
        AssertAlmostEqual(network.Response(ipt)[0], -0.221608, 1.0e-8, "Error in Network Response");
        delete ipt;
    }

    void test_multiple_responses(void) {
        //Tests the MLP_Network response method
        double *wgt;      // weights for each neuron
        double *ipt;      // Input for the network
        Perceptron *nrn;  // inserted neuron

        // Creating an empty network with 2 layers and 4 input dimensions
        bool biasc[] {false, true};

        MLP_Network network(2, 4, biasc);

        // Settup of the first network layer
            // First Neuron:
            wgt = new double[4] {0.1, -0.5, 0.3, 0.2};

            nrn = new LinearPerceptron(4, wgt, false);
            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 0);

            delete nrn;
            delete wgt;

            // Second Neuron:
            wgt = new double[4] {-0.1, -0.2, 0.4, -0.1};

            nrn = new LinearPerceptron(4, wgt, false);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 0);

            delete wgt;
            delete nrn;

        // Setup the second network layer
            // First Neuron:
            wgt = new double[3] {0.5, -0.7, 0.8};

            nrn = new LinearPerceptron(2, wgt, true);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 1);

            delete wgt;
            delete nrn;

        // Creating the input vector
        ipt = new double [4] {0.9694,  0.6909,  0.4334,  3.4965};
        AssertAlmostEqual(network.Response(ipt)[0], -0.221608, 1.0e-8, "Error in Multiple Responses");
        delete ipt;

        ipt = new double [4] { 0.5276,  0.0628,  0.9825,  0.3032};
        AssertAlmostEqual(network.Response(ipt)[0], -0.819777, 1.0e-8, "Error in Multiple Responses");
        delete ipt;

        ipt = new double [4] { 0.9153,  0.5782,  0.7620,  0.3580};
        AssertAlmostEqual(network.Response(ipt)[0], -0.791966, 1.0e-8, "Error in Multiple Responses");
        delete ipt;

        ipt = new double [4] { 0.9614,  0.4100,  0.2424, -3.4965};
        AssertAlmostEqual(network.Response(ipt)[0], -1.355649, 1.0e-8, "Error in Multiple Responses");
        delete ipt;

        ipt = new double [4] {-0.9694, -0.6909, -0.4334, -3.4965};
        AssertAlmostEqual(network.Response(ipt)[0], -1.378392, 1.0e-8, "Error in Multiple Responses");
        delete ipt;

        ipt = new double [4] { 0.9694,  0.6909,  0.4334,  3.4965};
        AssertAlmostEqual(network.Response(ipt)[0], -0.221608, 1.0e-8, "Error in Multiple Responses");
        delete ipt;
    }

    void test_sigmoidal_network(void) {
        //Tests the MLP_Network response method with a sigmoidal network
        double *wgt;      // weights for each neuron
        double *ipt;      // Input for the network
        Perceptron *nrn;  // inserted neuron

        // Creating an empty network with 2 layers and 4 input dimensions
        bool biasc[] {false, true};

        MLP_Network network(2, 4, biasc);

        // Settup of the first network layer
            // First Neuron:
            wgt = new double[4] {0.1, -0.5, 0.3, 0.2};

            nrn = new SigmoidalPerceptron(4, wgt, false, 1.0, 0.8);
            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 0);

            delete nrn;
            delete wgt;

            // Second Neuron:
            wgt = new double[4] {-0.1, -0.2, 0.4, -0.1};

            nrn = new RectifiedLinearPerceptron(4, wgt, false);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 0);

            delete wgt;
            delete nrn;

        // Setup the second network layer
            // First Neuron:
            wgt = new double[3] {0.5, -0.7, 0.8};

            nrn = new LinearPerceptron(2, wgt, true);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 1);

            delete wgt;
            delete nrn;

        // Creating the input vector
        ipt = new double [4] {0.9694,  0.6909,  0.4334,  3.4965};
        AssertAlmostEqual(network.Response(ipt)[0], -0.4929418849733654, 1.0e-8, "Error in Multiple Responses");
        delete ipt;

        ipt = new double [4] { 0.5276,  0.0628,  0.9825,  0.3032};
        AssertAlmostEqual(network.Response(ipt)[0], -0.7207596386362025, 1.0e-8, "Error in Multiple Responses");
        delete ipt;

        ipt = new double [4] { 0.9153,  0.5782,  0.7620,  0.3580};
        AssertAlmostEqual(network.Response(ipt)[0], -0.5830237614136182, 1.0e-8, "Error in Multiple Responses");
        delete ipt;

        ipt = new double [4] { 0.9614,  0.4100,  0.2424, -3.4965};
        AssertAlmostEqual(network.Response(ipt)[0], -0.8094224643476976, 1.0e-8, "Error in Multiple Responses");
        delete ipt;

        ipt = new double [4] {-0.9694, -0.6909, -0.4334, -3.4965};
        AssertAlmostEqual(network.Response(ipt)[0], -0.8950451150266348, 1.0e-8, "Error in Multiple Responses");
        delete ipt;

        ipt = new double [4] { 0.9694,  0.6909,  0.4334,  3.4965};
        AssertAlmostEqual(network.Response(ipt)[0], -0.4929418849733654, 1.0e-8, "Error in Multiple Responses");
        delete ipt;
    }

    void test_update_neuralweight_container(void) {
        //Tests the MLP_Network update weight container function
        double *wgt;      // weights for each neuron
        double *ipt;      // Input for the network
        Perceptron *nrn;  // inserted neuron

        // Creating an empty network with 2 layers and 4 input dimensions
        bool biasc[] {false, true};

        MLP_Network network(2, 4, biasc);

        // Settup of the first network layer
            // First Neuron:
            wgt = new double[4] {0.1, -0.5, 0.3, 0.2};

            nrn = new SigmoidalPerceptron(4, wgt, false, 1.0, 0.8);
            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 0);

            delete nrn;
            delete wgt;

            // Second Neuron:
            wgt = new double[4] {-0.1, -0.2, 0.4, -0.1};

            nrn = new RectifiedLinearPerceptron(4, wgt, false);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 0);

            delete wgt;
            delete nrn;

        // Setup the second network layer
            // First Neuron:
            wgt = new double[3] {0.5, -0.7, 0.8};

            nrn = new LinearPerceptron(2, wgt, true);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 1);

            delete wgt;
            delete nrn;

        AssertAlmostEqual(network.wghtcont()[0][0][2], 0.0, 1.0e-8, "Error in Update Neuralweight Container");
        network.Update_NeuralWeights_Container();
        AssertAlmostEqual(network.wghtcont()[0][0][2], 0.3, 1.0e-8, "Error in Update Neuralweight Container");
        AssertAlmostEqual(network.wghtcont()[1][0][1], -0.7, 1.0e-8, "Error in Update Neuralweight Container");
    }

    void test_sigmoidal_network_four_layers(void) {
        //Tests the MLP_Network with four layers
        double *wgt;      // weights for each neuron
        double *ipt;      // Input for the network
        Perceptron *nrn;  // inserted neuron

        // Creating an empty network with 4 layers and 2 input dimensions
        bool biasc[] {true, true, true, true};

        MLP_Network network(4, 2, biasc);

        // Settup of the first network layer
            // First Neuron:
            wgt = new double[3] {0.1, 0.2, 0.9};

            nrn = new SigmoidalPerceptron(2, wgt, true, 1.0, 0.8);
            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 0);

            delete nrn;
            delete wgt;

            // Second Neuron:
            wgt = new double[3] {0.4, -0.1, -0.2};

            nrn = new RectifiedLinearPerceptron(2, wgt, true);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 0);

            delete wgt;
            delete nrn;

        // Setup the second network layer
            // First Neuron:
            wgt = new double[3] {0.1, -0.9, -0.6};

            nrn = new SigmoidalPerceptron(2, wgt, true, 1.0, 0.65);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 1);

            delete wgt;
            delete nrn;

        // Setup the third network layer
            // First Neuron:
            wgt = new double[2] {0.1, -0.9};

            nrn = new SigmoidalPerceptron(1, wgt, true, 1.0, 0.7);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 2);

            delete wgt;
            delete nrn;

            // Second Neuron:
            wgt = new double[2] {-0.2, 0.9};

            nrn = new SigmoidalPerceptron(1, wgt, true, 1.0, 0.7);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 2);

            delete wgt;
            delete nrn;

            // Third Neuron:
            wgt = new double[2] {0.1, 0.1};

            nrn = new SigmoidalPerceptron(1, wgt, true, 1.0, 0.7);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 2);

            delete wgt;
            delete nrn;

            // Fourth Neuron:
            wgt = new double[2] {0.0, 0.0};

            nrn = new SigmoidalPerceptron(1, wgt, true, 1.0, 0.7);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 2);

            delete wgt;
            delete nrn;

        // Setup the last network layer
            // First Neuron:
            wgt = new double[5] {-0.5, 0.7, 0.9, -0.1, -0.2};

            nrn = new SigmoidalPerceptron(4, wgt, true, 1.0, 0.9);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 3);

            delete wgt;
            delete nrn;

        ipt = new double [2] {0.9694,  3.4965};
        AssertAlmostEqual(network.Response(ipt)[0], 0.6091912522089106, 1.0e-8, "Error in Network Response");
        delete ipt;

        ipt = new double [2] {0.5276,  0.3032};
        AssertAlmostEqual(network.Response(ipt)[0], 0.6092581052094788, 1.0e-8, "Error in Network Response");
        delete ipt;

        ipt = new double [2] {0.9153,  0.3580};
        AssertAlmostEqual(network.Response(ipt)[0], 0.6093219482014431, 1.0e-8, "Error in Network Response");
        delete ipt;

        ipt = new double [2] {0.9614, -3.4965};
        AssertAlmostEqual(network.Response(ipt)[0], 0.6095028730817934, 1.0e-8, "Error in Network Response");
        delete ipt;

        ipt = new double [2] {-0.9694, -3.4965};
        AssertAlmostEqual(network.Response(ipt)[0], 0.609171986992929, 1.0e-8, "Error in Network Response");
        delete ipt;

        ipt = new double [2] {0.9694,  3.4965};
        AssertAlmostEqual(network.Response(ipt)[0], 0.6091912522089106, 1.0e-8, "Error in Network Response");
        delete ipt;

        // Testing the update weight container
        AssertAlmostEqual(network.wghtcont()[0][1][0], 0.0, 1.0e-8, "Error in Update Neuralweight Container");
        AssertAlmostEqual(network.wghtcont()[1][0][2], 0.0, 1.0e-8, "Error in Update Neuralweight Container");
        AssertAlmostEqual(network.wghtcont()[2][2][1], 0.0, 1.0e-8, "Error in Update Neuralweight Container");
        AssertAlmostEqual(network.wghtcont()[3][0][4], 0.0, 1.0e-8, "Error in Update Neuralweight Container");
        network.Update_NeuralWeights_Container();
        AssertAlmostEqual(network.wghtcont()[0][1][0], 0.4, 1.0e-8, "Error in Update Neuralweight Container");
        AssertAlmostEqual(network.wghtcont()[1][0][2], -0.6, 1.0e-8, "Error in Update Neuralweight Container");
        AssertAlmostEqual(network.wghtcont()[2][2][1], 0.1, 1.0e-8, "Error in Update Neuralweight Container");
        AssertAlmostEqual(network.wghtcont()[3][0][4], -0.2, 1.0e-8, "Error in Update Neuralweight Container");

        // Testing the Activation Function container
        AssertAlmostEqual(network.actvfdercont()[0][0], 0.0, 1.0e-8, "Error in Update Activation Function Derivatives");
        AssertAlmostEqual(network.actvfdercont()[1][0], 0.0, 1.0e-8, "Error in Update Activation Function Derivatives");
        AssertAlmostEqual(network.actvfdercont()[2][3], 0.0, 1.0e-8, "Error in Update Activation Function Derivatives");
        AssertAlmostEqual(network.actvfdercont()[3][0], 0.0, 1.0e-8, "Error in Update Activation Function Derivatives");

        ipt = new double [2] {0.9694,  3.4965};
        network.Calculate_Network(ipt);
        delete ipt;

        AssertAlmostEqual(network.actvfdercont()[0][0], 0.1996558788494587, 1.0e-8, "Error in Update Activation Function Derivatives");
        AssertAlmostEqual(network.actvfdercont()[0][1], 1.0, 1.0e-8, "Error in Update Activation Function Derivatives");
        AssertAlmostEqual(network.actvfdercont()[1][0], 0.159314857434129, 1.0e-8, "Error in Update Activation Function Derivatives");
        AssertAlmostEqual(network.actvfdercont()[2][3], 0.175, 1.0e-8, "Error in Update Activation Function Derivatives");
        AssertAlmostEqual(network.actvfdercont()[3][0], 0.2142695433969451, 1.0e-8, "Error in Update Activation Function Derivatives");

        ipt = new double [2] {-0.7512,  -7.1862};
        network.Calculate_Network(ipt);
        delete ipt;

        AssertAlmostEqual(network.actvfdercont()[0][0], 0.08855529993623236, 1.0e-8, "Error in Update Activation Function Derivatives");
        AssertAlmostEqual(network.actvfdercont()[0][1], 1.0, 1.0e-8, "Error in Update Activation Function Derivatives");
        AssertAlmostEqual(network.actvfdercont()[1][0], 0.1624455101744401, 1.0e-8, "Error in Update Activation Function Derivatives");
        AssertAlmostEqual(network.actvfdercont()[2][3], 0.175, 1.0e-8, "Error in Update Activation Function Derivatives");
        AssertAlmostEqual(network.actvfdercont()[3][0], 0.21423419671674654, 1.0e-8, "Error in Update Activation Function Derivatives");

    }

    void test_network_three_outputs(void) {
        //Tests the MLP_Network with three neurons un last layer
        double *wgt;      // weights for each neuron
        double *ipt;      // Input for the network
        Perceptron *nrn;  // inserted neuron

        // Creating an empty network with 2 layers and 2 input dimensions
        bool biasc[] {true, true};

        MLP_Network network(2, 2, biasc);

        // Settup of the first network layer
            // First Neuron:
            wgt = new double[3] {0.1, 0.2, 0.9};

            nrn = new SigmoidalPerceptron(2, wgt, true, 1.0, 0.8);
            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 0);

            delete nrn;
            delete wgt;

            // Second Neuron:
            wgt = new double[3] {0.4, -0.1, -0.2};

            nrn = new RectifiedLinearPerceptron(2, wgt, true);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 0);

            delete wgt;
            delete nrn;

        // Setup the last network layer
            // First Neuron:
            wgt = new double[3] {0.1, -0.9, 0.8};

            nrn = new SigmoidalPerceptron(2, wgt, true, 1.0, 0.7);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 1);

            delete wgt;
            delete nrn;

            // Second Neuron:
            wgt = new double[3] {-0.2, 0.9, 0.6};

            nrn = new SigmoidalPerceptron(2, wgt, true, 1.0, 0.7);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 1);

            delete wgt;
            delete nrn;

            // Third Neuron:
            wgt = new double[3] {0.1, 0.1, -0.5};

            nrn = new SigmoidalPerceptron(2, wgt, true, 1.0, 0.7);

            // Adding the Neuron the network
            network.Insert_Neuron(nrn, 1);

            delete wgt;
            delete nrn;

        ipt = new double [2] {0.9694,  3.4965};
        AssertAlmostEqual(network.Response(ipt)[0], 0.3370516040817918, 1.0e-8, "Error in Network Response");
        AssertAlmostEqual(network.Response(ipt)[1], 0.4165172315165511, 1.0e-8, "Error in Network Response");
        AssertAlmostEqual(network.Response(ipt)[2], 0.5987395297459178, 1.0e-8, "Error in Network Response");
        delete ipt;

        ipt = new double [2] {-0.9694, -3.4965};
        AssertAlmostEqual(network.Response(ipt)[0], 0.3435171286906485, 1.0e-8, "Error in Network Response");
        AssertAlmostEqual(network.Response(ipt)[1], 0.4141898187335606, 1.0e-8, "Error in Network Response");
        AssertAlmostEqual(network.Response(ipt)[2], 0.5928267369344276, 1.0e-8, "Error in Network Response");
        delete ipt;

        ipt = new double [2] {0.9694,  3.4965};
        AssertAlmostEqual(network.Response(ipt)[0], 0.3370516040817918, 1.0e-8, "Error in Network Response");
        AssertAlmostEqual(network.Response(ipt)[1], 0.4165172315165511, 1.0e-8, "Error in Network Response");
        AssertAlmostEqual(network.Response(ipt)[2], 0.5987395297459178, 1.0e-8, "Error in Network Response");
        delete ipt;

        // Testing the update weight container
        AssertAlmostEqual(network.wghtcont()[0][1][0], 0.0, 1.0e-8, "Error in Update Neuralweight Container");
        AssertAlmostEqual(network.wghtcont()[1][0][2], 0.0, 1.0e-8, "Error in Update Neuralweight Container");
        network.Update_NeuralWeights_Container();
        AssertAlmostEqual(network.wghtcont()[0][1][0], 0.4, 1.0e-8, "Error in Update Neuralweight Container");
        AssertAlmostEqual(network.wghtcont()[1][0][2], 0.8, 1.0e-8, "Error in Update Neuralweight Container");

        // Testing the Activation Function container
        AssertAlmostEqual(network.actvfdercont()[0][0], 0.0, 1.0e-8, "Error in Update Activation Function Derivatives");
        AssertAlmostEqual(network.actvfdercont()[1][0], 0.0, 1.0e-8, "Error in Update Activation Function Derivatives");
        AssertAlmostEqual(network.actvfdercont()[1][2], 0.0, 1.0e-8, "Error in Update Activation Function Derivatives");

        ipt = new double [2] {0.9694,  3.4965};
        network.Calculate_Network(ipt);
        delete ipt;

        AssertAlmostEqual(network.actvfdercont()[0][0], 0.19965587884945870, 1.0e-8, "Error in Update Activation Function Derivatives");
        AssertAlmostEqual(network.actvfdercont()[1][0], 0.15641347418737797, 1.0e-8, "Error in Update Activation Function Derivatives");
        AssertAlmostEqual(network.actvfdercont()[1][2], 0.16817535368588848, 1.0e-8, "Error in Update Activation Function Derivatives");

        ipt = new double [2] {-0.7512,  -7.1862};
        network.Calculate_Network(ipt);
        delete ipt;

        AssertAlmostEqual(network.actvfdercont()[0][0], 0.08855529993623236, 1.0e-8, "Error in Update Activation Function Derivatives");
        AssertAlmostEqual(network.actvfdercont()[1][0], 0.14136235861291160, 1.0e-8, "Error in Update Activation Function Derivatives");
        AssertAlmostEqual(network.actvfdercont()[1][2], 0.16811120759014767, 1.0e-8, "Error in Update Activation Function Derivatives");

    }

};

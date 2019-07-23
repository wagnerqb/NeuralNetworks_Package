#include <iostream>

// Neuron Package Includes
#include "Test_Perceptron.h"
#include "Test_LinearPerceptron.h"
#include "Test_HeavySidePerceptron.h"
#include "Test_HyperbolicTangentPerceptron.h"
#include "Test_RectifiedLinearPerceptron.h"
#include "Test_SHLPerceptron.h"
#include "Test_SigmoidalPerceptron.h"
#include "Test_SymmetricLinearPerceptron.h"

// Network Package Includes
#include "Test_MLP_Network.h"

// Trainning Algorithms Package Includes
#include "Test_SumSquaresMean.h"
#include "Test_GeneralizedError.h"
#include "Test_TrainingAlgorithmsModel.h"
#include "Test_BackPropagation.h"
#include "Test_DescendingGradient.h"

/***************************************************************************//**
 * @class Test_NeuralNetPackage
 * @brief This class is the Unit Test for UnitTestPackage library
 * @ingroup Test_NeuralNetPackage
 ******************************************************************************/
class Test_NeuralNetPackage {

public:

    /*************************************************************************//**
     * Constructs a Test_NeuralNetPackage object
     ****************************************************************************/
    // Default constructor
    Test_NeuralNetPackage() {};
    virtual ~Test_NeuralNetPackage() {};     // Default destructor

    // Test Classes to be tested
    // Neurons Package
    Test_Perceptron tst_Perceptron;
    Test_LinearPerceptron tst_LinearPerceptron;
    Test_HeavySidePerceptron tst_HeavySidePerceptron;
    Test_HyperbolicTangentPerceptron tst_HyperbolicTangentPerceptron;
    Test_RectifiedLinearPerceptron tst_RectifiedLinearPerceptron;
    Test_SHLPerceptron tst_SHLPerceptron;
    Test_SigmoidalPerceptron tst_SigmoidalPerceptron;
    Test_SymmetricLinearPerceptron tst_SymmetricLinearPerceptron;

    // Network Package
    Test_MLP_Network tst_MLP_Network;

    // Trainning Algorithms Package
    Test_SumSquaresMean tst_SumSquaresMean;
    Test_GeneralizedError tst_GeneralizedError;
    Test_TrainingAlgorithmsModel tst_TrainingAlgorithmsModel;
    Test_BackPropagation tst_BackPropagation;
    Test_DescendingGradient tst_DescendingGradient;

    // Run_Tests
    void Run_Tests(void) {

        // Neuron Package Tests
        tst_Perceptron.Run_Tests();
        tst_LinearPerceptron.Run_Tests();
        tst_HeavySidePerceptron.Run_Tests();
        tst_HyperbolicTangentPerceptron.Run_Tests();
        tst_RectifiedLinearPerceptron.Run_Tests();
        tst_SHLPerceptron.Run_Tests();
        tst_SigmoidalPerceptron.Run_Tests();
        tst_SymmetricLinearPerceptron.Run_Tests();

        // Network Package Tests
        tst_MLP_Network.Run_Tests();

        // Trainning Algorithms Package Tests
        tst_SumSquaresMean.Run_Tests();
        tst_GeneralizedError.Run_Tests();
        tst_TrainingAlgorithmsModel.Run_Tests();
        tst_BackPropagation.Run_Tests();
        tst_DescendingGradient.Run_Tests();
    }

};

int main(int argc, char *argv[]) {

    std::cout << "Test NeuralNet Package\n" << std::endl;

    Test_NeuralNetPackage tst_pck;

    tst_pck.Run_Tests();

    std::cout << "NeuralNet Test Performed, Press Enter to Exit.\n";
    std::cin.get();

  return 0;
}

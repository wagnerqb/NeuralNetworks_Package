#include <iostream>
#include "UnitTestPackage.h"
#include "MLP_Network.h"
#include "TrainingAlgorithmsModel.h"
#include "BackPropagation.h"

/***************************************************************************//**
 * @class Test_BackPropagation
 * @brief This class is the Unit Test for BackPropagation Class
 * @ingroup Test_NeuralNetPackage
 ******************************************************************************/
class Test_BackPropagation : public UnitTestPackage {

public:

    /*************************************************************************//**
     * Constructs a Test_BackPropagation object
     ****************************************************************************/
    // Default constructor
    Test_BackPropagation() {
        UnitTestPackage();
    };

    // Default destructor
    virtual ~Test_BackPropagation() {

    };

    // Run_Tests
    void Run_Tests(void) {

        test_update_error_derivatives();

        Generate_TestLog("BackPropagation");
    }

    // Tests
    void test_update_error_derivatives(void) {
        //Tests the BackPropagation with three neurons un last layer
        double *wgt;      // weights for each neuron
        Perceptron *nrn;  // inserted neuron
        TrainingAlgorithmsModel *TrainingModel;  // Model for Training Algorithm

        // Step 1: Preparing Trainning Data:
        std::vector < double * > tdt_;
        tdt_ = Prepare_TrainningData();

        // Step 2: Preparing Network
        bool biasc[] {true, true};
        MLP_Network network(2, 3, biasc);

        // layercfg = (10, 1)
        std::vector < double * > weights_fst;
        weights_fst.push_back( new double[4] { 0.20,  0.15, -0.10,  0.50});
        weights_fst.push_back( new double[4] { 0.18, -0.15,  0.10, -0.15});
        weights_fst.push_back( new double[4] { 0.02,  0.11, -0.14,  0.25});
        weights_fst.push_back( new double[4] {-0.20,  0.01,  0.10, -0.65});
        weights_fst.push_back( new double[4] { 0.53,  0.85, -0.41,  0.95});
        weights_fst.push_back( new double[4] { 0.44, -0.10, -0.81, -0.75});
        weights_fst.push_back( new double[4] {-0.26, -0.88, -0.01,  0.15});
        weights_fst.push_back( new double[4] { 0.10,  0.10,  0.10,  0.10});
        weights_fst.push_back( new double[4] { 0.00,  0.00,  0.00,  0.00});
        weights_fst.push_back( new double[4] {-0.12,  0.15, -0.10, -0.45});

        // Settup First Layer
        for (int ftl_ = 0; ftl_ < 10; ftl_++) {
            nrn = new SigmoidalPerceptron(3, weights_fst[ftl_], biasc[0], 1.0, 1.0);
            network.Insert_Neuron(nrn, 0);
            delete nrn;
        }

        // Settup Last Layer
        wgt = new double[11] {0.4105, 0.695, 1.000, 0.4321, 0.8404, 0.0036, 0.194, 0.3274, 0.2697, 0.265, 0.0161};
        nrn = new LinearPerceptron(10, wgt, biasc[1], 1.0);
        network.Insert_Neuron(nrn, 1);
        delete nrn;
        delete wgt;

        TrainingModel = new BackPropagation(&network);

        // TrainingModel->Training(tdt_, 0.2, 1000);
        TrainingModel->Training(tdt_, 0.2, 1000, 1.0e-6, 1.0e-8, 0.25, 100, true, true);

        // Network Responses
        double * ipt;

        ipt = new double [4] {0.0611, 0.2860, 0.7464};
        AssertAlmostEqual(network.Response(ipt)[0], 0.50405109, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.5102, 0.7464, 0.0860};
        AssertAlmostEqual(network.Response(ipt)[0], 0.60344726, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.0004, 0.6916, 0.5006};
        AssertAlmostEqual(network.Response(ipt)[0], 0.54764470, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.9430, 0.4476, 0.2648};
        AssertAlmostEqual(network.Response(ipt)[0], 0.71793883, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.1399, 0.1610, 0.2477};
        AssertAlmostEqual(network.Response(ipt)[0], 0.28098580, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.6423, 0.3229, 0.8567};
        AssertAlmostEqual(network.Response(ipt)[0], 0.76873579, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.6492, 0.0007, 0.6422};
        AssertAlmostEqual(network.Response(ipt)[0], 0.58203987, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.1818, 0.5078, 0.9046};
        AssertAlmostEqual(network.Response(ipt)[0], 0.68849071, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.7382, 0.2647, 0.1916};
        AssertAlmostEqual(network.Response(ipt)[0], 0.55494118, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.3879, 0.1307, 0.8656};
        AssertAlmostEqual(network.Response(ipt)[0], 0.61072972, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.1903, 0.6523, 0.7820};
        AssertAlmostEqual(network.Response(ipt)[0], 0.70482783, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.8401, 0.4490, 0.2719};
        AssertAlmostEqual(network.Response(ipt)[0], 0.68885384, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.0029, 0.3264, 0.2476};
        AssertAlmostEqual(network.Response(ipt)[0], 0.29214821, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.7088, 0.9342, 0.2763};
        AssertAlmostEqual(network.Response(ipt)[0], 0.79473148, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.1283, 0.1882, 0.7253};
        AssertAlmostEqual(network.Response(ipt)[0], 0.48372960, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.8882, 0.3077, 0.8931};
        AssertAlmostEqual(network.Response(ipt)[0], 0.84523065, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.2225, 0.9182, 0.7820};
        AssertAlmostEqual(network.Response(ipt)[0], 0.79714028, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.1957, 0.8423, 0.3085};
        AssertAlmostEqual(network.Response(ipt)[0], 0.60585798, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.9991, 0.5914, 0.3933};
        AssertAlmostEqual(network.Response(ipt)[0], 0.82052017, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;
        ipt = new double [4] {0.2299, 0.1524, 0.7353};
        AssertAlmostEqual(network.Response(ipt)[0], 0.51444192, 1.0e-8, "Error in TrainingModel Response");
        delete ipt;

        // Cleaning created data
        for (int dt_=0; dt_ < 200; dt_++) {
            delete tdt_[dt_];
        }

        for (int ftl_=0; ftl_ < 10; ftl_++) {
            delete weights_fst[ftl_];
        }

    }

    std::vector < double * > Prepare_TrainningData(void){

        std::vector < double * > tdt_;

        tdt_.push_back( new double[4] {0.8799, 0.7998, 0.3972, 0.8399});
        tdt_.push_back( new double[4] {0.57, 0.5111, 0.2418, 0.6258});
        tdt_.push_back( new double[4] {0.6796, 0.4117, 0.337, 0.6622});
        tdt_.push_back( new double[4] {0.3567, 0.2967, 0.6037, 0.5969});
        tdt_.push_back( new double[4] {0.3866, 0.839, 0.0232, 0.5316});
        tdt_.push_back( new double[4] {0.0271, 0.7788, 0.7445, 0.6335});
        tdt_.push_back( new double[4] {0.8174, 0.8422, 0.3229, 0.8068});
        tdt_.push_back( new double[4] {0.6027, 0.1468, 0.3759, 0.5342});
        tdt_.push_back( new double[4] {0.1203, 0.326, 0.5419, 0.4768});
        tdt_.push_back( new double[4] {0.1325, 0.2082, 0.4934, 0.4105});
        tdt_.push_back( new double[4] {0.695, 1.0, 0.4321, 0.8404});
        tdt_.push_back( new double[4] {0.0036, 0.194, 0.3274, 0.2697});
        tdt_.push_back( new double[4] {0.265, 0.0161, 0.5947, 0.4125});
        tdt_.push_back( new double[4] {0.5849, 0.6019, 0.4376, 0.7464});
        tdt_.push_back( new double[4] {0.0108, 0.3538, 0.181, 0.28});
        tdt_.push_back( new double[4] {0.9008, 0.7264, 0.9184, 0.9602});
        tdt_.push_back( new double[4] {0.0023, 0.9659, 0.3182, 0.4986});
        tdt_.push_back( new double[4] {0.1366, 0.6357, 0.6967, 0.6459});
        tdt_.push_back( new double[4] {0.8621, 0.7353, 0.2742, 0.7718});
        tdt_.push_back( new double[4] {0.0682, 0.9624, 0.4211, 0.5764});
        tdt_.push_back( new double[4] {0.6112, 0.6014, 0.5254, 0.7868});
        tdt_.push_back( new double[4] {0.003, 0.7585, 0.8928, 0.6388});
        tdt_.push_back( new double[4] {0.7644, 0.5964, 0.0407, 0.6055});
        tdt_.push_back( new double[4] {0.6441, 0.2097, 0.5847, 0.6545});
        tdt_.push_back( new double[4] {0.0803, 0.3799, 0.602, 0.4991});
        tdt_.push_back( new double[4] {0.1908, 0.8046, 0.5402, 0.6665});
        tdt_.push_back( new double[4] {0.6937, 0.3967, 0.6055, 0.7595});
        tdt_.push_back( new double[4] {0.2591, 0.0582, 0.3978, 0.3604});
        tdt_.push_back( new double[4] {0.4241, 0.185, 0.9066, 0.6298});
        tdt_.push_back( new double[4] {0.3332, 0.9303, 0.2475, 0.6287});
        tdt_.push_back( new double[4] {0.3625, 0.1592, 0.9981, 0.5948});
        tdt_.push_back( new double[4] {0.9259, 0.096, 0.1645, 0.4716});
        tdt_.push_back( new double[4] {0.8606, 0.6779, 0.0033, 0.6242});
        tdt_.push_back( new double[4] {0.0838, 0.5472, 0.3758, 0.4835});
        tdt_.push_back( new double[4] {0.0303, 0.9191, 0.7233, 0.6491});
        tdt_.push_back( new double[4] {0.9293, 0.8319, 0.9664, 0.984});
        tdt_.push_back( new double[4] {0.7268, 0.144, 0.9753, 0.7096});
        tdt_.push_back( new double[4] {0.2888, 0.6593, 0.4078, 0.6328});
        tdt_.push_back( new double[4] {0.5515, 0.1364, 0.2894, 0.4745});
        tdt_.push_back( new double[4] {0.7683, 0.0067, 0.5546, 0.5708});
        tdt_.push_back( new double[4] {0.6462, 0.6761, 0.834, 0.8933});
        tdt_.push_back( new double[4] {0.3694, 0.2212, 0.1233, 0.3658});
        tdt_.push_back( new double[4] {0.2706, 0.3222, 0.9996, 0.631});
        tdt_.push_back( new double[4] {0.6282, 0.1404, 0.8474, 0.6733});
        tdt_.push_back( new double[4] {0.5861, 0.6693, 0.3818, 0.7433});
        tdt_.push_back( new double[4] {0.6057, 0.9901, 0.5141, 0.8466});
        tdt_.push_back( new double[4] {0.5915, 0.5588, 0.3055, 0.6787});
        tdt_.push_back( new double[4] {0.8359, 0.4145, 0.5016, 0.7597});
        tdt_.push_back( new double[4] {0.5497, 0.6319, 0.8382, 0.8521});
        tdt_.push_back( new double[4] {0.7072, 0.1721, 0.3812, 0.5772});
        tdt_.push_back( new double[4] {0.1185, 0.5084, 0.8376, 0.6211});
        tdt_.push_back( new double[4] {0.6365, 0.5562, 0.4965, 0.7693});
        tdt_.push_back( new double[4] {0.4145, 0.5797, 0.8599, 0.7878});
        tdt_.push_back( new double[4] {0.2575, 0.5358, 0.4028, 0.5777});
        tdt_.push_back( new double[4] {0.2026, 0.33, 0.3054, 0.4261});
        tdt_.push_back( new double[4] {0.3385, 0.0476, 0.5941, 0.4625});
        tdt_.push_back( new double[4] {0.4094, 0.1726, 0.7803, 0.6015});
        tdt_.push_back( new double[4] {0.1261, 0.6181, 0.4927, 0.5739});
        tdt_.push_back( new double[4] {0.1224, 0.4662, 0.2146, 0.4007});
        tdt_.push_back( new double[4] {0.6793, 0.6774, 1.0, 0.9141});
        tdt_.push_back( new double[4] {0.8176, 0.0358, 0.2506, 0.4707});
        tdt_.push_back( new double[4] {0.6937, 0.6685, 0.5075, 0.822});
        tdt_.push_back( new double[4] {0.2404, 0.5411, 0.8754, 0.698});
        tdt_.push_back( new double[4] {0.6553, 0.2609, 0.1188, 0.4851});
        tdt_.push_back( new double[4] {0.8886, 0.0288, 0.2604, 0.4802});
        tdt_.push_back( new double[4] {0.3974, 0.5275, 0.6457, 0.7215});
        tdt_.push_back( new double[4] {0.2108, 0.491, 0.5432, 0.5913});
        tdt_.push_back( new double[4] {0.8675, 0.5571, 0.1849, 0.6805});
        tdt_.push_back( new double[4] {0.5693, 0.0242, 0.9293, 0.6033});
        tdt_.push_back( new double[4] {0.8439, 0.4631, 0.6345, 0.8226});
        tdt_.push_back( new double[4] {0.3644, 0.2948, 0.3937, 0.524});
        tdt_.push_back( new double[4] {0.2014, 0.6326, 0.9782, 0.7143});
        tdt_.push_back( new double[4] {0.4039, 0.0645, 0.4629, 0.4547});
        tdt_.push_back( new double[4] {0.7137, 0.067, 0.2359, 0.4602});
        tdt_.push_back( new double[4] {0.4277, 0.9555, 0.0, 0.5477});
        tdt_.push_back( new double[4] {0.0259, 0.7634, 0.2889, 0.4738});
        tdt_.push_back( new double[4] {0.1871, 0.7682, 0.9697, 0.7397});
        tdt_.push_back( new double[4] {0.3216, 0.542, 0.0677, 0.4526});
        tdt_.push_back( new double[4] {0.2524, 0.7688, 0.9523, 0.7711});
        tdt_.push_back( new double[4] {0.3621, 0.5295, 0.2521, 0.5571});
        tdt_.push_back( new double[4] {0.2942, 0.1625, 0.2745, 0.3759});
        tdt_.push_back( new double[4] {0.818, 0.0023, 0.1439, 0.4018});
        tdt_.push_back( new double[4] {0.8429, 0.1704, 0.5251, 0.6563});
        tdt_.push_back( new double[4] {0.9612, 0.6898, 0.663, 0.9128});
        tdt_.push_back( new double[4] {0.1009, 0.419, 0.0826, 0.3055});
        tdt_.push_back( new double[4] {0.7071, 0.7704, 0.8328, 0.9298});
        tdt_.push_back( new double[4] {0.3371, 0.7819, 0.0959, 0.5377});
        tdt_.push_back( new double[4] {0.1555, 0.5599, 0.9221, 0.6663});
        tdt_.push_back( new double[4] {0.7318, 0.1877, 0.3311, 0.5689});
        tdt_.push_back( new double[4] {0.1665, 0.7449, 0.0997, 0.4508});
        tdt_.push_back( new double[4] {0.8762, 0.2498, 0.9167, 0.7829});
        tdt_.push_back( new double[4] {0.9885, 0.6229, 0.2085, 0.72});
        tdt_.push_back( new double[4] {0.0461, 0.7745, 0.5632, 0.5949});
        tdt_.push_back( new double[4] {0.3209, 0.6229, 0.5233, 0.681});
        tdt_.push_back( new double[4] {0.9189, 0.593, 0.7288, 0.8989});
        tdt_.push_back( new double[4] {0.0382, 0.5515, 0.8818, 0.5999});
        tdt_.push_back( new double[4] {0.3726, 0.9988, 0.3814, 0.7086});
        tdt_.push_back( new double[4] {0.4211, 0.2668, 0.3307, 0.508});
        tdt_.push_back( new double[4] {0.2378, 0.0817, 0.3574, 0.3452});
        tdt_.push_back( new double[4] {0.9893, 0.7637, 0.2526, 0.7755});
        tdt_.push_back( new double[4] {0.8203, 0.0682, 0.426, 0.5643});
        tdt_.push_back( new double[4] {0.6226, 0.2146, 0.1021, 0.4452});
        tdt_.push_back( new double[4] {0.4589, 0.3147, 0.2236, 0.4962});
        tdt_.push_back( new double[4] {0.3471, 0.8889, 0.1564, 0.5875});
        tdt_.push_back( new double[4] {0.5762, 0.8292, 0.4116, 0.7853});
        tdt_.push_back( new double[4] {0.9053, 0.6245, 0.5264, 0.8506});
        tdt_.push_back( new double[4] {0.286, 0.0793, 0.0549, 0.2224});
        tdt_.push_back( new double[4] {0.9567, 0.3034, 0.4425, 0.6993});
        tdt_.push_back( new double[4] {0.517, 0.9266, 0.1565, 0.6594});
        tdt_.push_back( new double[4] {0.8149, 0.0396, 0.6227, 0.6165});
        tdt_.push_back( new double[4] {0.371, 0.3554, 0.5633, 0.6171});
        tdt_.push_back( new double[4] {0.8702, 0.3185, 0.2762, 0.6287});
        tdt_.push_back( new double[4] {0.1016, 0.6382, 0.3173, 0.4957});
        tdt_.push_back( new double[4] {0.389, 0.2369, 0.0083, 0.3235});
        tdt_.push_back( new double[4] {0.2702, 0.8617, 0.1218, 0.5319});
        tdt_.push_back( new double[4] {0.7473, 0.6507, 0.5582, 0.8464});
        tdt_.push_back( new double[4] {0.9108, 0.2139, 0.4641, 0.6625});
        tdt_.push_back( new double[4] {0.4343, 0.6028, 0.1344, 0.5546});
        tdt_.push_back( new double[4] {0.6847, 0.4062, 0.9318, 0.8204});
        tdt_.push_back( new double[4] {0.8657, 0.9448, 0.99, 0.9904});
        tdt_.push_back( new double[4] {0.4011, 0.4138, 0.8715, 0.7222});
        tdt_.push_back( new double[4] {0.5949, 0.26, 0.081, 0.448});
        tdt_.push_back( new double[4] {0.1845, 0.7906, 0.9725, 0.7425});
        tdt_.push_back( new double[4] {0.3438, 0.6725, 0.9821, 0.7926});
        tdt_.push_back( new double[4] {0.8398, 0.136, 0.9119, 0.7222});
        tdt_.push_back( new double[4] {0.2245, 0.0971, 0.6136, 0.4402});
        tdt_.push_back( new double[4] {0.3742, 0.9668, 0.8194, 0.8371});
        tdt_.push_back( new double[4] {0.9572, 0.9836, 0.3793, 0.8556});
        tdt_.push_back( new double[4] {0.7496, 0.041, 0.136, 0.4059});
        tdt_.push_back( new double[4] {0.9123, 0.351, 0.0682, 0.5455});
        tdt_.push_back( new double[4] {0.6954, 0.55, 0.6801, 0.8388});
        tdt_.push_back( new double[4] {0.5252, 0.6529, 0.5729, 0.7893});
        tdt_.push_back( new double[4] {0.3156, 0.3851, 0.5983, 0.6161});
        tdt_.push_back( new double[4] {0.146, 0.1637, 0.0249, 0.1813});
        tdt_.push_back( new double[4] {0.778, 0.4491, 0.4614, 0.7498});
        tdt_.push_back( new double[4] {0.5959, 0.8647, 0.8601, 0.9176});
        tdt_.push_back( new double[4] {0.2204, 0.1785, 0.4607, 0.4276});
        tdt_.push_back( new double[4] {0.7355, 0.8264, 0.7015, 0.9214});
        tdt_.push_back( new double[4] {0.9931, 0.6727, 0.3139, 0.7829});
        tdt_.push_back( new double[4] {0.9123, 0.0, 0.1106, 0.3944});
        tdt_.push_back( new double[4] {0.2858, 0.9688, 0.2262, 0.5988});
        tdt_.push_back( new double[4] {0.7931, 0.8993, 0.9028, 0.9728});
        tdt_.push_back( new double[4] {0.7841, 0.0778, 0.9012, 0.6832});
        tdt_.push_back( new double[4] {0.138, 0.5881, 0.2367, 0.4622});
        tdt_.push_back( new double[4] {0.6345, 0.5165, 0.7139, 0.8191});
        tdt_.push_back( new double[4] {0.2453, 0.5888, 0.1559, 0.4765});
        tdt_.push_back( new double[4] {0.1174, 0.5436, 0.3657, 0.4953});
        tdt_.push_back( new double[4] {0.3667, 0.3228, 0.6952, 0.6376});
        tdt_.push_back( new double[4] {0.9532, 0.6949, 0.4451, 0.8426});
        tdt_.push_back( new double[4] {0.7954, 0.8346, 0.0449, 0.6676});
        tdt_.push_back( new double[4] {0.1427, 0.048, 0.6267, 0.378});
        tdt_.push_back( new double[4] {0.1516, 0.9824, 0.0827, 0.4627});
        tdt_.push_back( new double[4] {0.4868, 0.6223, 0.7462, 0.8116});
        tdt_.push_back( new double[4] {0.3408, 0.5115, 0.0783, 0.4559});
        tdt_.push_back( new double[4] {0.8146, 0.6378, 0.5837, 0.8628});
        tdt_.push_back( new double[4] {0.282, 0.5409, 0.7256, 0.6939});
        tdt_.push_back( new double[4] {0.5716, 0.2958, 0.5477, 0.6619});
        tdt_.push_back( new double[4] {0.9323, 0.0229, 0.4797, 0.5731});
        tdt_.push_back( new double[4] {0.2907, 0.7245, 0.5165, 0.6911});
        tdt_.push_back( new double[4] {0.0068, 0.0545, 0.0861, 0.0851});
        tdt_.push_back( new double[4] {0.2636, 0.9885, 0.2175, 0.5847});
        tdt_.push_back( new double[4] {0.035, 0.3653, 0.7801, 0.5117});
        tdt_.push_back( new double[4] {0.967, 0.3031, 0.7127, 0.7836});
        tdt_.push_back( new double[4] {0.0, 0.7763, 0.8735, 0.6388});
        tdt_.push_back( new double[4] {0.4395, 0.0501, 0.9761, 0.5712});
        tdt_.push_back( new double[4] {0.9359, 0.0366, 0.9514, 0.6826});
        tdt_.push_back( new double[4] {0.0173, 0.9548, 0.4289, 0.5527});
        tdt_.push_back( new double[4] {0.6112, 0.907, 0.6286, 0.8803});
        tdt_.push_back( new double[4] {0.201, 0.9573, 0.6791, 0.7283});
        tdt_.push_back( new double[4] {0.8914, 0.9144, 0.2641, 0.7966});
        tdt_.push_back( new double[4] {0.0061, 0.0802, 0.8621, 0.3711});
        tdt_.push_back( new double[4] {0.2212, 0.4664, 0.3821, 0.526});
        tdt_.push_back( new double[4] {0.2401, 0.6964, 0.0751, 0.4637});
        tdt_.push_back( new double[4] {0.7881, 0.9833, 0.3038, 0.8049});
        tdt_.push_back( new double[4] {0.2435, 0.0794, 0.5551, 0.4223});
        tdt_.push_back( new double[4] {0.2752, 0.8414, 0.2797, 0.6079});
        tdt_.push_back( new double[4] {0.7616, 0.4698, 0.5337, 0.7809});
        tdt_.push_back( new double[4] {0.3395, 0.0022, 0.0087, 0.1836});
        tdt_.push_back( new double[4] {0.7849, 0.9981, 0.4449, 0.8641});
        tdt_.push_back( new double[4] {0.8312, 0.0961, 0.2129, 0.4857});
        tdt_.push_back( new double[4] {0.9763, 0.1102, 0.6227, 0.6667});
        tdt_.push_back( new double[4] {0.8597, 0.3284, 0.6932, 0.7829});
        tdt_.push_back( new double[4] {0.9295, 0.3275, 0.7536, 0.8016});
        tdt_.push_back( new double[4] {0.2435, 0.2163, 0.7625, 0.5449});
        tdt_.push_back( new double[4] {0.9281, 0.8356, 0.5285, 0.8991});
        tdt_.push_back( new double[4] {0.8313, 0.7566, 0.6192, 0.9047});
        tdt_.push_back( new double[4] {0.1712, 0.0545, 0.5033, 0.3561});
        tdt_.push_back( new double[4] {0.0609, 0.1702, 0.4306, 0.331});
        tdt_.push_back( new double[4] {0.5899, 0.9408, 0.0369, 0.6245});
        tdt_.push_back( new double[4] {0.7858, 0.5115, 0.0916, 0.6066});
        tdt_.push_back( new double[4] {1.0, 0.1653, 0.7103, 0.7172});
        tdt_.push_back( new double[4] {0.2007, 0.1163, 0.3431, 0.3385});
        tdt_.push_back( new double[4] {0.2306, 0.033, 0.0293, 0.159});
        tdt_.push_back( new double[4] {0.8477, 0.6378, 0.4623, 0.8254});
        tdt_.push_back( new double[4] {0.9677, 0.7895, 0.9467, 0.9782});
        tdt_.push_back( new double[4] {0.0339, 0.4669, 0.1526, 0.325});
        tdt_.push_back( new double[4] {0.008, 0.8988, 0.4201, 0.5404});
        tdt_.push_back( new double[4] {0.9955, 0.8897, 0.6175, 0.936});
        tdt_.push_back( new double[4] {0.7408, 0.5351, 0.2732, 0.6949});
        tdt_.push_back( new double[4] {0.6843, 0.3737, 0.1562, 0.5625});

        return tdt_;
    }

};

/***************************************************************************//**
 * @file     UnitTestPackage.cpp
 * @date     28 Mar 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup UnitTestPackage UnitTestPackage
 * @brief    This class is responsable for create and manage
             unit tests, focused in scientific applications.
 ******************************************************************************/

#include "UnitTestPackage.h"
#include <iostream>

// Default constructor
UnitTestPackage::UnitTestPackage() {

    // Initializing class
    number_of_tests_ = 0;  ///< Number of realized tests
    number_of_errors_ = 0; ///< Number of failed tests

};

// AssertIntEqual
void UnitTestPackage::AssertIntEqual(int int_1, int int_2, std::string error_message) {

    std::string err_msg;
    bool assert_result;

    if (error_message == "Default Message") {
        err_msg = "int_1 != int_2";
    } else {
        err_msg = error_message;
    }


    //assert_result = AssertRealFunctions.AssertEqual(var_1, var_2);
    assert_result = (int_1 == int_2);
    number_of_tests_++;

        if (not assert_result){

            number_of_errors_++;

            std::cout <<  "***********************************************************\n";
            std::cout <<  err_msg << std::endl;
            std::cout <<  int_1 << " != " << int_2 << std::endl;
            std::cout <<  "***********************************************************" << std::endl;
        }

};

// AssertDoubleEqual
void UnitTestPackage::AssertDoubleEqual(double double_1, double double_2, std::string error_message) {

    std::string err_msg;
    bool assert_result;

    if (error_message == "Default Message") {
        err_msg = "double_1 != double_2";
    } else {
        err_msg = error_message;
    }


    //assert_result = AssertRealFunctions.AssertEqual(var_1, var_2);
    assert_result = (double_1 == double_2);
    number_of_tests_++;

        if (not assert_result){

            number_of_errors_++;

            std::cout <<  "***********************************************************\n";
            std::cout <<  err_msg << std::endl;
            std::cout <<  double_1 << " != " << double_2 << std::endl;
            std::cout <<  "***********************************************************" << std::endl;
        }

};

// AssertDoubleEqual
void UnitTestPackage::AssertAlmostEqual(double double_1, double double_2, double eps, std::string error_message) {

    std::string err_msg;
    double difference;
    bool assert_result;

    if (error_message == "Default Message") {
        err_msg = "double_1 != double_2";
    } else {
        err_msg = error_message;
    }

    difference = fabs(double_1 - double_2);
    assert_result = (difference <= eps);
    number_of_tests_++;

        if (not assert_result){

            number_of_errors_++;

            std::cout <<  "***********************************************************\n";
            std::cout <<  err_msg << std::endl;
            std::cout <<  double_1 << " != " << double_2 << std::endl;
            std::cout <<  "***********************************************************" << std::endl;
        }

};

// Generate_TestLog
void UnitTestPackage::Generate_TestLog(std::string class_name) {

    std::string error_log_;
    std::string class_name_;

    if (class_name == "Default Class") {
        class_name_ = "Some Class";
    } else {
        class_name_ = class_name;
    }

    error_log_ = "***********************************************************\n";
    error_log_ += "UnitTest Generated for Class " + class_name_ + "\n";
    error_log_ += "Number of Performed Tests: " + std::to_string(number_of_tests_) + "\n";
    error_log_ += "Number of Failed Tests: " + std::to_string(number_of_errors_) + "\n";
    error_log_ += "Execution End.\n";
    error_log_ += "***********************************************************\n";

    std::cout << error_log_ << std::endl;
};

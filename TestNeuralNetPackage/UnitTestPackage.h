/***************************************************************************//**
 * @file     UnitTestPackage.h
 * @date     28 Mar 2019
 * @author   Wagner Queiroz Barros <wagnerqb@gmail.com>
 * @defgroup UnitTestPackage UnitTestPackage
 * @brief    This class is responsable for create and manage
             unit tests, focused in scientific applications.
 ******************************************************************************/

#ifndef UNITTESTPACKAGE_H
#define UNITTESTPACKAGE_H

#include <string>
#include <cmath>

/***************************************************************************//**
 * @class UnitTestPackage
 * @brief This class is responsable for create and manage
          unit tests, focused in scientific applications.
 * @ingroup UnitTestPackage
 ******************************************************************************/
class UnitTestPackage {

public:

    /*************************************************************************//**
     * Constructs a UnitTestPackage object
     ****************************************************************************/
    UnitTestPackage();                 // Default constructor
    virtual ~UnitTestPackage() {};     // Default destructor

    // getters
    inline int number_of_tests() const { return number_of_tests_; }
    inline int number_of_errors() const { return number_of_errors_; }

    // AssertIntEqual
    /*************************************************************************//**
     * @brief Function that assert that int_1 is exactly the same of
     *        int_2
     * @param int_1 The first variable of comparison
     * @param int_1 The second variable of comparison
     * @param error_msg Error message returned in case of failure
     ****************************************************************************/
    void AssertIntEqual(int int_1, int int_2, std::string error_msg="Default Message");

    // AssertDoubleEqual
    /*************************************************************************//**
     * @brief Function that assert that double_1 is exactly the same of
     *        double_2
     * @param double_1 The first variable of comparison
     * @param double_1 The second variable of comparison
     * @param error_msg Error message returned in case of failure
     ****************************************************************************/
    void AssertDoubleEqual(double double_1, double double_2, std::string error_msg="Default Message");

    // AssertAlmostEqual
    /*************************************************************************//**
     * @brief Function that assert that double_1 is almost the same of
     *        double_2
     * @param double_1 The first variable of comparison
     * @param double_1 The second variable of comparison
     * @param    eps   Tolerance for difference between double_1 and double_2
     * @param error_msg Error message returned in case of failure
     ****************************************************************************/
    void AssertAlmostEqual(double double_1, double double_2, double eps=1.0e-6, std::string error_msg="Default Message");

    // Generate_TestLog
    /*************************************************************************//**
     * @brief Function that generate the log of all executed tests
     ****************************************************************************/
    void Generate_TestLog(std::string class_name="Default Class");


protected:
    int number_of_tests_;  ///< Number of realized tests
    int number_of_errors_; ///< Number of failed tests

private:

};


#endif


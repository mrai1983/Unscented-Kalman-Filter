#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0,0,0,0;

    // Checking the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size

    if(estimations.size() != ground_truth.size()
       || estimations.empty())
    {
        cout << "Invalid estimation or ground_truth data" << endl;
        return rmse;
    }

    //accumulate squared residuals
    for(unsigned int i=0; i < estimations.size(); ++i)
    {

        VectorXd residual = estimations[i] - ground_truth[i];

        //coefficient-wise multiplication
        residual = residual.array()*residual.array();
        rmse += residual;
    }

    rmse = rmse/ estimations.size();

    rmse = rmse.array().sqrt();

    //return the result
    return rmse;
}



void Tools::ProcessNIS(VectorXd &z_diff, MatrixXd &S, MeasurementPackage meas_package)
{
    int ret = (z_diff.transpose() * S.inverse() * z_diff);

    if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
        cout << "NIS_radar = " << ret << endl;
        ofstream out("NisRadar.txt", ios::app);
        out << ret << endl;
        out.close();
    }
    else
    {
        cout << "NIS_lidar = " << ret << endl;
        ofstream out("NisLidar.txt", ios::app);
        out << ret << endl;
        out.close();
    }
}

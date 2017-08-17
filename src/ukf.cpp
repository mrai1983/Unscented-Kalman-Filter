#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_,n_x_);

  P_ << 0.15, 0, 0, 0, 0,
		0, 0.15, 0, 0, 0,
		0, 0, 0.15, 0, 0,
		0, 0, 0, 0.15, 0,
		0, 0, 0, 0, 0.15;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;


  is_initialized_ = false;

  use_laser_ = false;

  use_radar_ = false;

  n_aug_ = 7;

  lambda_ = 3 - n_x_;

  time_us_ = 0;

  //Matrix for weights
  weights_ = VectorXd(2 * n_aug_ + 1);

  //Matrix holding generated sigma points.
  x_sig_ = MatrixXd(n_aug_, 2 * n_aug_ + 1 );

  //Matrix holding augmented states
  x_aug_ = VectorXd(n_aug_);

  //Matrix for augmented covariance
  p_aug_ = MatrixXd(n_aug_, n_aug_);

  //Matrix to hold predicted sigmapoints in prediction
  //space
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

	if (is_initialized_ == false)
	{
		//Initialize state matrix and time stamp and return
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
		{
			//For radar we receive ro, rodot and theta. Hence px
			double ro = meas_package.raw_measurements_[0];
			double phi = meas_package.raw_measurements_[1];
			double rodot = meas_package.raw_measurements_[2];

			x_ <<   ro * sin(phi),
					ro * cos(phi),
					2,
					2,
					2;
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
		{
			x_ <<   meas_package.raw_measurements_[0],
					meas_package.raw_measurements_[1],
					2,
					2,
					2;
		}

		time_us_ = meas_package.timestamp_;

		is_initialized_ = true;

		return;
	}

	double dt = (meas_package.timestamp_ - time_us_)/1000000.0;
	time_us_ = meas_package.timestamp_;

	PredictSigmaPoints(dt);


	/***********************************************************************************
	 * Map predicted mean and covariance in sensor space
	 ***********************************************************************************/

	if (meas_package.sensor_type_ == MeasurementPackage::LASER)
	{
		UpdateLidar(meas_package);
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
	{
		//UpdateRadar(meas_package);
	}

}


void UKF::GenerateSigmaPoints()
{
	/********************************************************************
	 * Generate augmented sigma points
	 *******************************************************************/
	MatrixXd x_sig = MatrixXd(n_aug_, 2 * n_aug_ + 1 );

	VectorXd x_aug = VectorXd(n_aug_);

	MatrixXd p_aug = MatrixXd(n_aug_, n_aug_);

	x_aug.head(5) = x_;
	x_aug(5) = 0;
	x_aug(6) = 0;

	p_aug.fill(0.0);

	p_aug.topLeftCorner(5,5) = P_;

	p_aug(5,5) = std_a_ * std_a_;
	p_aug(6,6) = std_yawdd_ * std_yawdd_;

	MatrixXd L = p_aug.llt().matrixL();

	for (int i = 0; i < n_aug_; i++)
	{
		x_sig.col(i+1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
		x_sig.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
	}
}

void UKF::PredictSigmaPoints(double dt)
{
	/*******************************************************************
	 * Sigma points are generated in 7x15 matrix. Now predict new
	 * sigma points using x = f(x+v).
	 *******************************************************************/
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{
		double p_x = x_sig_(0, i);
		double p_y = x_sig_(1, i);
		double v = x_sig_(2, i);
		double yaw = x_sig_(3,i);
		double yawd = x_sig_(4,i);
		double nu_a = x_sig_(5, i);
		double nu_yawdd = x_sig_(6, i);

		double px_p = 0.0;
		double py_p = 0.0;

		//Check for divide by zero
	    if (fabs(yawd) > 0.001)
	    {
	        px_p = p_x + v/yawd * ( sin (yaw + yawd*dt) - sin(yaw));
	        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*dt) );
	    }
	    else
	    {
	        px_p = p_x + v*dt*cos(yaw);
	        py_p = p_y + v*dt*sin(yaw);
	    }

	    double v_p = v;
	    double yaw_p = yaw + yaw * dt;
	    double yawd_p = yawd;

	    //Adding noise
	    px_p = px_p + 0.5 * nu_a * dt * dt * cos(yaw);
	    py_p = py_p + 0.5 * nu_a * dt * dt * sin(yaw);

	    v_p = v_p + nu_a * dt;

	    yaw_p = yaw_p + 0.5 * nu_yawdd * dt * dt;
	    yawd_p = yawd_p + nu_yawdd * dt;

	    Xsig_pred_(0,i) = px_p;
	    Xsig_pred_(1,i) = py_p;
	    Xsig_pred_(2,i) = v_p;
	    Xsig_pred_(3,i) = yaw_p;
	    Xsig_pred_(4,i) = yawd_p;

	}

}

void UKF::PredictStateAndCovariance()
{

	/*****************************************************************************
	 * Mean and Covariance Prediction
	 ****************************************************************************/

	double weight_0 = lambda_/(lambda_+n_aug_);
	weights_(0) = weight_0;

	for (int i = 1; i < 2*n_aug_ + 1; i++)
	{
		double weight = 0.5/(n_aug_ + lambda_);
		weights_(i) = weight;
	}

	x_.fill(0.0);

	for (int i = 0; i < 2* n_aug_ + 1; i++)
	{
		x_ = x_ + weights_(i) * Xsig_pred_.col(i);
	}

	P_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		while (x_diff(3) > M_PI)
		{
			x_diff(3)-=2.*M_PI;
		}

		while (x_diff(3) < -M_PI)
		{
			x_diff(3)+=2.*M_PI;
		}

		P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
	}
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
	PredictSigmaPoints(delta_t);
	PredictStateAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

	int n_z = 2;

	MatrixXd zSig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//Transform predicted sigma points into measurement space.
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{

		//Radar data contains ro, theta and rodot.
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);

		zSig(0,i) = p_x;
		zSig(1,i) = p_y;
	}

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);
	z_pred.fill(0.0);

	for (int i=0; i < 2*n_aug_+1; i++)
	{
		z_pred = z_pred + weights_(i) * zSig.col(i);
	}

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{
		VectorXd z_diff = zSig.col(i) - z_pred;

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	//add measurement noise covariance matrix
	MatrixXd R = MatrixXd(n_z, n_z);

	R <<    std_laspx_*std_laspx_, 0,
		    0, std_laspx_*std_laspx_;


	S = S + R;


	//create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);

	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{
		VectorXd z_diff = zSig.col(i) - z_pred;

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	VectorXd z = VectorXd(n_z);

	z << meas_package.raw_measurements_[0],
		 meas_package.raw_measurements_[1];



	//Kalman gain K;
	MatrixXd K = Tc * S.inverse();

	//residual
	VectorXd z_diff = z - z_pred;

	//update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K*S*K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	/**
	TODO:

	Complete this function! Use radar data to update the belief about the object's
	position. Modify the state vector, x_, and covariance, P_.

	You'll also need to calculate the radar NIS.
	*/

	int n_z = 3;

	MatrixXd zSig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//Transform predicted sigma points into measurement space.
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{

		//Radar data contains ro, theta and rodot.
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);

	    double v1 = cos(yaw)*v;
	    double v2 = sin(yaw)*v;

		zSig(0,i) = sqrt (p_x* p_x + p_y*p_y);
		zSig(1,i) = atan2 (p_y,p_x);
		zSig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);
	}

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);
	z_pred.fill(0.0);
	for (int i=0; i < 2*n_aug_+1; i++)
	{
		z_pred = z_pred + weights_(i) * zSig.col(i);
	}

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{
		VectorXd z_diff = zSig.col(i) - z_pred;

		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	//add measurement noise covariance matrix
	MatrixXd R = MatrixXd(n_z, n_z);

	R <<    std_radr_*std_radr_, 0, 0,
		    0, std_radphi_*std_radphi_, 0,
		    0, 0,std_radrd_*std_radrd_;

	S = S + R;


	//create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);

	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++)
	{
		VectorXd z_diff = zSig.col(i) - z_pred;

		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	VectorXd z = VectorXd(n_z);

	z << meas_package.raw_measurements_[0],
			meas_package.raw_measurements_[1],
			meas_package.raw_measurements_[2];


	  //Kalman gain K;
	  MatrixXd K = Tc * S.inverse();

	  //residual
	  VectorXd z_diff = z - z_pred;

	  //angle normalization
	  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
	  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

	  //update state mean and covariance matrix
	  x_ = x_ + K * z_diff;
	  P_ = P_ - K*S*K.transpose();

}

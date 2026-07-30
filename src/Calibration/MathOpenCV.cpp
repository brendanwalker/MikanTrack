#include "MathOpenCV.h"

bool opencv_point2f_compute_average(const std::vector<cv::Point2f>& points, cv::Point2f& outAveragePoint)
{
	bool success= false;
	const int count= (int)points.size();

	if (count == 1)
	{
		outAveragePoint= points[0];
		success= true;
	}
	else if (count > 1)
	{
		outAveragePoint= cv::Point2f(0, 0);
		for (int i= 0; i < count; ++i)
		{
			const cv::Point2f& v= points[i];

			outAveragePoint+= v;
		}
		outAveragePoint/= (double)points.size();
		success= true;
	}

	return success;
}

bool opencv_vec3d_compute_average(const std::vector<cv::Vec3d>& vectors, cv::Vec3d& outAverageVec)
{
	bool success= false;
	const int count= (int)vectors.size();

	if (count == 1)
	{
		outAverageVec= vectors[0];
		success= true;
	}
	else if (count > 1)
	{
		outAverageVec= cv::Vec3d(0, 0, 0);
		for (int i= 0; i < count; ++i)
		{
			const cv::Vec3d& v= vectors[i];

			outAverageVec+= v;
		}
		outAverageVec/= (double)vectors.size();
		success= true;
	}

	return success;
}

bool opencv_quaternion_compute_average(const std::vector<cv::Quatd>& quaternions, cv::Quatd& outAverageQuat)
{
	bool success= false;
	const int count= (int)quaternions.size();

	if (count == 1)
	{
		outAverageQuat= quaternions[0];
		success= true;
	}
	else if (count == 2)
	{
		outAverageQuat= cv::Quatd::nlerp(quaternions[0], quaternions[1], 0.5);
		success= true;
	}
	else if (count > 2)
	{
		// http://stackoverflow.com/questions/12374087/average-of-multiple-quaternions
		// Paper: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf
		cv::Mat q(4, count, CV_64FC1);
		cv::Mat q_transpose(count, 4, CV_64FC1);

		const double weight= 1.0 / (double)count;
		for (int index= 0; index < count; ++index)
		{
			// Normalize the weights against the total weight
			const cv::Quatd& sample= quaternions[index];

			const double w= sample.w * weight;
			const double x= sample.x * weight;
			const double y= sample.y * weight;
			const double z= sample.z * weight;

			q.at<double>(0, index)= w;
			q.at<double>(1, index)= x;
			q.at<double>(2, index)= y;
			q.at<double>(3, index)= z;

			q_transpose.at<double>(index, 0)= w;
			q_transpose.at<double>(index, 1)= x;
			q_transpose.at<double>(index, 2)= y;
			q_transpose.at<double>(index, 3)= z;
		}

		cv::Mat M= q * q_transpose;

		//    src*eigenvectors.row(i).t() = eigenvalues.at<srcType>(i)*eigenvectors.row(i).t()
		cv::Mat eigenValues, eigenVectors;
		if (cv::eigen(M, eigenValues, eigenVectors))
		{
			// The first Eigen vector corresponds to the largest Eigen value
			const cv::Vec4d largestEigenVector= eigenVectors.row(0);

			const float w= (float)largestEigenVector(0);
			const float x= (float)largestEigenVector(1);
			const float y= (float)largestEigenVector(2);
			const float z= (float)largestEigenVector(3);

			outAverageQuat= cv::Quatd(w, x, y, z).normalize();
			success= true;
		}
	}

	return success;
}
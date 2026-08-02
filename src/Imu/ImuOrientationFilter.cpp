#include "ImuOrientationFilter.h"

#include <algorithm>
#include <cmath>

#include "glm/geometric.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/mat3x3.hpp"

static constexpr float k_gravityMetersPerSecond2= 9.80665f;
// World "up" - the tracking world is Z-up (marker plane normal)
static const glm::vec3 k_worldUp(0.f, 0.f, 1.f);

namespace
{
// -- tiny fixed-size linear algebra (row-major) -----
using Mat6= std::array<float, 36>;
using Mat63= std::array<float, 18>; // 6 rows x 3 cols
using Mat3= std::array<float, 9>;

inline float& at6(Mat6& m, int row, int col) { return m[row * 6 + col]; }
inline float at6(const Mat6& m, int row, int col) { return m[row * 6 + col]; }
inline float& at63(Mat63& m, int row, int col) { return m[row * 3 + col]; }
inline float at63(const Mat63& m, int row, int col) { return m[row * 3 + col]; }
inline float& at3(Mat3& m, int row, int col) { return m[row * 3 + col]; }
inline float at3(const Mat3& m, int row, int col) { return m[row * 3 + col]; }

Mat3 skewSymmetric(const glm::vec3& v)
{
	return Mat3{0.f, -v.z, v.y, v.z, 0.f, -v.x, -v.y, v.x, 0.f};
}

Mat3 identity3()
{
	return Mat3{1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
}

// P= F * P * F^T  (F given as four 3x3 blocks)
Mat6 propagateCovariance(const Mat6& covariance, const Mat3& fThetaTheta, const Mat3& fThetaBias)
{
	// F= [ fThetaTheta  fThetaBias ]
	//    [ 0            I          ]
	Mat6 f{};
	for (int row= 0; row < 3; ++row)
	{
		for (int col= 0; col < 3; ++col)
		{
			at6(f, row, col)= at3(fThetaTheta, row, col);
			at6(f, row, col + 3)= at3(fThetaBias, row, col);
			at6(f, row + 3, col)= 0.f;
			at6(f, row + 3, col + 3)= row == col ? 1.f : 0.f;
		}
	}

	Mat6 fp{};
	for (int row= 0; row < 6; ++row)
	{
		for (int col= 0; col < 6; ++col)
		{
			float sum= 0.f;
			for (int k= 0; k < 6; ++k)
				sum+= at6(f, row, k) * at6(covariance, k, col);
			at6(fp, row, col)= sum;
		}
	}

	Mat6 result{};
	for (int row= 0; row < 6; ++row)
	{
		for (int col= 0; col < 6; ++col)
		{
			float sum= 0.f;
			for (int k= 0; k < 6; ++k)
				sum+= at6(fp, row, k) * at6(f, col, k); // * F^T
			at6(result, row, col)= sum;
		}
	}
	return result;
}

bool invert3x3(const Mat3& m, Mat3& outInverse)
{
	const float a= at3(m, 0, 0), b= at3(m, 0, 1), c= at3(m, 0, 2);
	const float d= at3(m, 1, 0), e= at3(m, 1, 1), f= at3(m, 1, 2);
	const float g= at3(m, 2, 0), h= at3(m, 2, 1), i= at3(m, 2, 2);

	const float determinant= a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
	if (fabsf(determinant) < 1e-20f)
		return false;

	const float inverseDeterminant= 1.f / determinant;
	outInverse= Mat3{(e * i - f * h) * inverseDeterminant, (c * h - b * i) * inverseDeterminant,
					 (b * f - c * e) * inverseDeterminant, (f * g - d * i) * inverseDeterminant,
					 (a * i - c * g) * inverseDeterminant, (c * d - a * f) * inverseDeterminant,
					 (d * h - e * g) * inverseDeterminant, (b * g - a * h) * inverseDeterminant,
					 (a * e - b * d) * inverseDeterminant};
	return true;
}
} // namespace

void ImuOrientationFilter::configure(const ImuOrientationFilterConfig& config)
{
	m_config= config;
	reset();
}

void ImuOrientationFilter::reset()
{
	m_orientation= glm::quat(1.f, 0.f, 0.f, 0.f);
	m_gyroBias= glm::vec3(0.f);
	m_lastAngularVelocity= glm::vec3(0.f);
	m_bInitialized= false;

	m_covariance.fill(0.f);
	// Tilt (body X/Y) is pinned quickly by gravity; yaw (body Z) is not
	// observable from inertial data at all, so it starts wide open
	at6(m_covariance, 0, 0)= m_config.initialTiltSigma * m_config.initialTiltSigma;
	at6(m_covariance, 1, 1)= m_config.initialTiltSigma * m_config.initialTiltSigma;
	at6(m_covariance, 2, 2)= m_config.initialYawSigma * m_config.initialYawSigma;
	for (int axis= 3; axis < 6; ++axis)
		at6(m_covariance, axis, axis)= m_config.initialBiasSigma * m_config.initialBiasSigma;
}

void ImuOrientationFilter::initializeFromGravity(const glm::vec3& acceleration)
{
	const float magnitude= glm::length(acceleration);
	if (magnitude < 1e-3f)
		return;

	// Rotate the measured "up" (specific force direction) onto world up. This
	// leaves yaw at identity, which is correct - it's unobservable here.
	const glm::vec3 measuredUp= acceleration / magnitude;
	const float alignment= glm::dot(measuredUp, k_worldUp);
	if (alignment > 0.999999f)
	{
		m_orientation= glm::quat(1.f, 0.f, 0.f, 0.f);
	}
	else if (alignment < -0.999999f)
	{
		// Antiparallel: any perpendicular axis is a valid 180-degree flip
		m_orientation= glm::angleAxis(3.14159265f, glm::vec3(1.f, 0.f, 0.f));
	}
	else
	{
		const glm::vec3 axis= glm::normalize(glm::cross(measuredUp, k_worldUp));
		m_orientation= glm::normalize(glm::angleAxis(acosf(std::clamp(alignment, -1.f, 1.f)), axis));
	}
	m_bInitialized= true;
}

void ImuOrientationFilter::predict(const glm::vec3& angularVelocity, float dtSeconds)
{
	// Guard against timestamp glitches / stalls
	const float dt= std::clamp(dtSeconds, 1e-5f, 0.1f);

	const glm::vec3 correctedRate= angularVelocity - m_gyroBias;
	m_lastAngularVelocity= correctedRate;

	// Nominal orientation: body-frame integration q <- q * exp(w dt / 2)
	const float angle= glm::length(correctedRate) * dt;
	if (angle > 1e-9f)
	{
		const glm::vec3 axis= correctedRate / glm::length(correctedRate);
		m_orientation= glm::normalize(m_orientation * glm::angleAxis(angle, axis));
	}

	// Error-state transition (body-frame error parameterization):
	//   dtheta' = (I - [w]x dt) dtheta - I dt * dbias
	//   dbias'  = dbias
	const Mat3 skew= skewSymmetric(correctedRate);
	Mat3 fThetaTheta= identity3();
	Mat3 fThetaBias{};
	for (int i= 0; i < 9; ++i)
	{
		fThetaTheta[i]-= skew[i] * dt;
		fThetaBias[i]= 0.f;
	}
	at3(fThetaBias, 0, 0)= -dt;
	at3(fThetaBias, 1, 1)= -dt;
	at3(fThetaBias, 2, 2)= -dt;

	m_covariance= propagateCovariance(m_covariance, fThetaTheta, fThetaBias);

	// Process noise: angle random walk on orientation, random walk on bias
	const float orientationNoise= m_config.gyroNoiseDensity * m_config.gyroNoiseDensity * dt;
	const float biasNoise= m_config.gyroBiasRandomWalk * m_config.gyroBiasRandomWalk * dt;
	for (int axis= 0; axis < 3; ++axis)
	{
		at6(m_covariance, axis, axis)+= orientationNoise;
		at6(m_covariance, axis + 3, axis + 3)+= biasNoise;
	}
}

void ImuOrientationFilter::applyCorrection(const glm::vec3& residual, const Mat3& hTheta, const Mat3& hBias,
										   const Mat3& measurementNoise)
{
	// H is 3x6 = [hTheta | hBias]
	auto hAt= [&hTheta, &hBias](int row, int col) -> float {
		return col < 3 ? at3(hTheta, row, col) : at3(hBias, row, col - 3);
	};

	// PHt = P * H^T  (6x3)
	Mat63 pht{};
	for (int row= 0; row < 6; ++row)
	{
		for (int col= 0; col < 3; ++col)
		{
			float sum= 0.f;
			for (int k= 0; k < 6; ++k)
				sum+= at6(m_covariance, row, k) * hAt(col, k);
			at63(pht, row, col)= sum;
		}
	}

	// S = H * PHt + R  (3x3)
	Mat3 innovationCovariance{};
	for (int row= 0; row < 3; ++row)
	{
		for (int col= 0; col < 3; ++col)
		{
			float sum= 0.f;
			for (int k= 0; k < 6; ++k)
				sum+= hAt(row, k) * at63(pht, k, col);
			at3(innovationCovariance, row, col)= sum + at3(measurementNoise, row, col);
		}
	}

	Mat3 innovationInverse{};
	if (!invert3x3(innovationCovariance, innovationInverse))
		return;

	// K = PHt * S^-1  (6x3)
	Mat63 gain{};
	for (int row= 0; row < 6; ++row)
	{
		for (int col= 0; col < 3; ++col)
		{
			float sum= 0.f;
			for (int k= 0; k < 3; ++k)
				sum+= at63(pht, row, k) * at3(innovationInverse, k, col);
			at63(gain, row, col)= sum;
		}
	}

	// Error state dx = K * residual
	float errorState[6]= {};
	for (int row= 0; row < 6; ++row)
	{
		errorState[row]= at63(gain, row, 0) * residual.x + at63(gain, row, 1) * residual.y +
			at63(gain, row, 2) * residual.z;
	}

	// Joseph form: P = (I-KH) P (I-KH)^T + K R K^T (stays symmetric positive
	// definite even with imperfect arithmetic, which matters for a filter
	// that runs for hours)
	Mat6 ikh{};
	for (int row= 0; row < 6; ++row)
	{
		for (int col= 0; col < 6; ++col)
		{
			float sum= row == col ? 1.f : 0.f;
			for (int k= 0; k < 3; ++k)
				sum-= at63(gain, row, k) * hAt(k, col);
			at6(ikh, row, col)= sum;
		}
	}

	Mat6 temp{};
	for (int row= 0; row < 6; ++row)
	{
		for (int col= 0; col < 6; ++col)
		{
			float sum= 0.f;
			for (int k= 0; k < 6; ++k)
				sum+= at6(ikh, row, k) * at6(m_covariance, k, col);
			at6(temp, row, col)= sum;
		}
	}

	Mat6 updated{};
	for (int row= 0; row < 6; ++row)
	{
		for (int col= 0; col < 6; ++col)
		{
			float sum= 0.f;
			for (int k= 0; k < 6; ++k)
				sum+= at6(temp, row, k) * at6(ikh, col, k); // * (I-KH)^T
			at6(updated, row, col)= sum;
		}
	}
	// + K R K^T
	for (int row= 0; row < 6; ++row)
	{
		for (int col= 0; col < 6; ++col)
		{
			float sum= 0.f;
			for (int a= 0; a < 3; ++a)
			{
				for (int b= 0; b < 3; ++b)
					sum+= at63(gain, row, a) * at3(measurementNoise, a, b) * at63(gain, col, b);
			}
			at6(updated, row, col)+= sum;
		}
	}
	m_covariance= updated;

	// Inject the error into the nominal state, then the error is zero again
	const glm::vec3 orientationError(errorState[0], errorState[1], errorState[2]);
	const float errorAngle= glm::length(orientationError);
	if (errorAngle > 1e-9f)
	{
		m_orientation=
			glm::normalize(m_orientation * glm::angleAxis(errorAngle, orientationError / errorAngle));
	}
	m_gyroBias+= glm::vec3(errorState[3], errorState[4], errorState[5]);
}

bool ImuOrientationFilter::updateWithGravity(const glm::vec3& acceleration)
{
	const float magnitude= glm::length(acceleration);
	if (magnitude < 1e-3f)
		return false;

	// Gate: only believe the accelerometer when it plausibly reads gravity
	// alone. Under real motion the specific force is gravity PLUS linear
	// acceleration, and using it would drag the tilt estimate around.
	if (fabsf(magnitude - k_gravityMetersPerSecond2) > m_config.accelGate)
		return false;

	if (!m_bInitialized)
	{
		initializeFromGravity(acceleration);
		return true;
	}

	// Predicted gravity direction in SENSOR frame: R^T * worldUp
	const glm::mat3 sensorToWorld= glm::mat3_cast(m_orientation);
	const glm::vec3 predictedUp= glm::transpose(sensorToWorld) * k_worldUp;
	const glm::vec3 measuredUp= acceleration / magnitude;

	// h(q) = R^T * up; with body-frame error, dh/dtheta = [R^T up]x
	const Mat3 hTheta= skewSymmetric(predictedUp);
	Mat3 hBias{}; // gravity says nothing about bias directly

	const float sigma= m_config.accelNoise / k_gravityMetersPerSecond2; // normalized units
	Mat3 measurementNoise{};
	at3(measurementNoise, 0, 0)= sigma * sigma;
	at3(measurementNoise, 1, 1)= sigma * sigma;
	at3(measurementNoise, 2, 2)= sigma * sigma;

	applyCorrection(measuredUp - predictedUp, hTheta, hBias, measurementNoise);
	return true;
}

void ImuOrientationFilter::updateWithOrientation(const glm::quat& sensorToWorld)
{
	if (!m_bInitialized)
	{
		m_orientation= glm::normalize(sensorToWorld);
		m_bInitialized= true;
		return;
	}

	// Residual = log(q_nominal^-1 * q_measured), a body-frame rotation vector
	glm::quat errorQuaternion= glm::normalize(glm::inverse(m_orientation) * glm::normalize(sensorToWorld));
	if (errorQuaternion.w < 0.f) // shortest arc
		errorQuaternion= -errorQuaternion;

	const glm::vec3 errorAxis(errorQuaternion.x, errorQuaternion.y, errorQuaternion.z);
	const float axisLength= glm::length(errorAxis);
	glm::vec3 residual(0.f);
	if (axisLength > 1e-9f)
	{
		const float angle= 2.f * atan2f(axisLength, errorQuaternion.w);
		residual= errorAxis / axisLength * angle;
	}

	const Mat3 hTheta= identity3(); // measures the orientation error directly
	Mat3 hBias{};

	Mat3 measurementNoise{};
	const float sigma= m_config.visionNoise;
	at3(measurementNoise, 0, 0)= sigma * sigma;
	at3(measurementNoise, 1, 1)= sigma * sigma;
	at3(measurementNoise, 2, 2)= sigma * sigma;

	applyCorrection(residual, hTheta, hBias, measurementNoise);
}

void ImuOrientationFilter::processSample(const ImuSample& sample, float dtSeconds)
{
	if (!m_bInitialized)
		initializeFromGravity(sample.acceleration);

	predict(sample.angularVelocity, dtSeconds);
	updateWithGravity(sample.acceleration);
}

glm::vec3 ImuOrientationFilter::getOrientationSigma() const
{
	return glm::vec3(sqrtf(std::max(at6(m_covariance, 0, 0), 0.f)),
					 sqrtf(std::max(at6(m_covariance, 1, 1), 0.f)),
					 sqrtf(std::max(at6(m_covariance, 2, 2), 0.f)));
}

bool ImuOrientationFilter::isTiltConverged() const
{
	// Tilt is the body X/Y error; 3 degrees is comfortably converged
	constexpr float kConvergedSigma= 0.05f; // rad
	const glm::vec3 sigma= getOrientationSigma();
	return m_bInitialized && sigma.x < kConvergedSigma && sigma.y < kConvergedSigma;
}

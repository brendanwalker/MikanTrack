#include "TestCommon.h"

// WHAT THIS CANNOT TELL YOU: it compares the gyro against the
// accelerometer WITHIN one device, so it is blind to any error in the
// sensor frame as a whole. Acceleration is a vector and angular
// velocity is a pseudovector, so under a reflection both pick up sign
// flips that cancel in dg/dt = -w x g - a mirrored frame satisfies the
// correct physical relation and passes cleanly.
//
// Settling a frame convention needs an EXTERNAL reference. The one
// that worked was strapping both controllers rigidly together and
// solving for the transform between their raw streams: identical
// physical motion, no vision, no mounting, no wrist, and the
// determinant answers rotation-vs-reflection outright. Chasing it
// through vision instead produced three inconclusive analyses and two
// confidently wrong corrections.
// Solves the fixed transform between TWO rigidly coupled IMUs from a
// dump's raw sample history, and reports whether it is a rotation or a
// reflection.
//
// This is the measurement that settles a sensor frame convention, and
// it is here as a tool because the alternative - inferring it from
// vision - produced three inconclusive analyses and two confidently
// wrong corrections before a five-minute rigid-body capture answered
// it outright. Strap both controllers to one object, tumble it through
// all three axes for a few seconds, press F9, and run this on the dump.
//
// det(Q) = +1 : the frames differ by a ROTATION, which the mounting
//               calibration absorbs. No correction is needed.
// det(Q) = -1 : a REFLECTION, which no quaternion mounting can
//               represent, so it must be corrected in the raw samples.
//
// The scale ratio comes free and is worth reading: on one rigid body
// both gyros must report the same rate, so a ratio far from 1 is a
// scale-factor problem that no axis remapping would ever fix.
static int runImuPairTest(const TestArgs& args)
{
	if (args.empty())
	{
		MIKAN_LOG_ERROR("test-imupair") << "usage: --test-imupair <path to dump.json>";
		return 1;
	}

	int result= 0;
	try
	{
		std::ifstream dumpFile(args[0]);
		if (!dumpFile.is_open())
		{
			MIKAN_LOG_ERROR("test-imupair") << "Could not open " << args[0];
			return 1;
		}
		nlohmann::json dump;
		dumpFile >> dump;

		auto readSide= [&dump](const char* side, std::vector<double>& outTimes,
							   std::vector<cv::Vec3d>& outGyro, std::vector<cv::Vec3d>& outAccel) {
			for (const nlohmann::json& sample : dump["imuRaw"][side])
			{
				outTimes.push_back(sample["t"].get<double>());
				outGyro.push_back(cv::Vec3d(sample["gyro"][0], sample["gyro"][1], sample["gyro"][2]));
				outAccel.push_back(
					cv::Vec3d(sample["accel"][0], sample["accel"][1], sample["accel"][2]));
			}
		};

		std::vector<double> leftTimes, rightTimes;
		std::vector<cv::Vec3d> leftGyro, rightGyro, leftAccel, rightAccel;
		readSide("left", leftTimes, leftGyro, leftAccel);
		readSide("right", rightTimes, rightGyro, rightAccel);

		if (leftTimes.size() < 50 || rightTimes.size() < 50)
		{
			MIKAN_LOG_ERROR("test-imupair")
				<< "Need raw samples from BOTH controllers (left " << leftTimes.size() << ", right "
				<< rightTimes.size() << ")";
			return 1;
		}

		// Resample the right onto the left's clock; both are the same
		// steady_clock so this is interpolation, not alignment
		auto sampleAt= [](const std::vector<double>& times, const std::vector<cv::Vec3d>& values,
						  double t) {
			const size_t index= std::min(
				std::max<size_t>(
					(size_t)(std::lower_bound(times.begin(), times.end(), t) - times.begin()), 1),
				times.size() - 1);
			const double span= std::max(1e-9, times[index] - times[index - 1]);
			const double alpha= std::clamp((t - times[index - 1]) / span, 0.0, 1.0);
			return values[index - 1] * (1.0 - alpha) + values[index] * alpha;
		};

		struct PairStats
		{
			const char* name;
			double det= 0.0;
			double residualDegrees= 0.0;
			double scaleRatio= 0.0;
			int samples= 0;
			cv::Matx33d transform;
		};

		auto solve= [&](const char* name, const std::vector<cv::Vec3d>& leftValues,
						const std::vector<cv::Vec3d>& rightValues, double minMagnitude) {
			PairStats stats;
			stats.name= name;

			cv::Matx33d covariance= cv::Matx33d::zeros();
			std::vector<std::pair<cv::Vec3d, cv::Vec3d>> pairs;
			double ratioSum= 0.0;
			for (size_t index= 0; index < leftTimes.size(); ++index)
			{
				const double t= leftTimes[index];
				if (t < rightTimes.front() || t > rightTimes.back())
					continue;
				const cv::Vec3d a= leftValues[index];
				const cv::Vec3d b= sampleAt(rightTimes, rightValues, t);
				const double na= cv::norm(a), nb= cv::norm(b);
				if (na < minMagnitude || nb < minMagnitude)
					continue;
				pairs.emplace_back(a, b);
				ratioSum+= nb / na;
				// H = sum b a^T; the transform maps LEFT into RIGHT
				for (int row= 0; row < 3; ++row)
					for (int col= 0; col < 3; ++col)
						covariance(row, col)+= b[row] * a[col];
			}
			stats.samples= (int)pairs.size();
			if (stats.samples < 30)
				return stats;

			stats.scaleRatio= ratioSum / (double)stats.samples;

			cv::Mat w, u, vt;
			cv::SVD::compute(cv::Mat(covariance), w, u, vt, cv::SVD::FULL_UV);
			const cv::Matx33d transform= cv::Matx33d((cv::Mat)(u * vt));
			stats.transform= transform;
			stats.det= cv::determinant(transform);

			double errorSum= 0.0;
			for (const std::pair<cv::Vec3d, cv::Vec3d>& pair : pairs)
			{
				const cv::Vec3d mapped= transform * (pair.first / cv::norm(pair.first));
				const double dot= std::clamp(mapped.dot(pair.second / cv::norm(pair.second)), -1.0, 1.0);
				errorSum+= std::acos(dot);
			}
			stats.residualDegrees= errorSum / (double)pairs.size() * 180.0 / CV_PI;
			return stats;
		};

		const PairStats gyro= solve("gyro", leftGyro, rightGyro, 0.5);
		const PairStats accel= solve("accel", leftAccel, rightAccel, 1.0);

		for (const PairStats& stats : {gyro, accel})
		{
			if (stats.samples < 30)
			{
				MIKAN_LOG_ERROR("test-imupair")
					<< stats.name << ": only " << stats.samples
					<< " usable samples - tumble the pair through more motion";
				result= 1;
				continue;
			}
			MIKAN_LOG_INFO("test-imupair")
				<< stats.name << ": det=" << stats.det << " ("
				<< (stats.det > 0.0 ? "ROTATION - the mounting absorbs it, no correction needed"
									: "REFLECTION - must be corrected in the raw samples")
				<< "), residual=" << stats.residualDegrees << " deg, right/left magnitude ratio="
				<< stats.scaleRatio << ", n=" << stats.samples;
		}

		// The two sensors must agree about which it is; disagreeing
		// means one of the two streams is not what it claims to be
		if (gyro.samples >= 30 && accel.samples >= 30 &&
			(gyro.det > 0.0) != (accel.det > 0.0))
		{
			MIKAN_LOG_ERROR("test-imupair")
				<< "gyro and accelerometer disagree on handedness - one stream is mislabelled";
			result= 1;
		}
	}
	catch (const std::exception& e)
	{
		MIKAN_LOG_ERROR("test-imupair") << "Failed to analyze the dump: " << e.what();
		result= 1;
	}

	return result;
}

MIKAN_REGISTER_TEST("--test-imupair", "Solves the transform between two rigidly coupled IMUs from a dump", eTestCategory::Tool, runImuPairTest);

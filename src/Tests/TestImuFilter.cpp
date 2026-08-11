#include "TestCommon.h"

static int runImuFilterTest(const TestArgs& args)
{
	int result= 0;

	constexpr float kGravity= 9.80665f;
	const glm::vec3 worldUp(0.f, 0.f, 1.f);
	constexpr float kDt= 1.f / 200.f; // Joy-Con rate

	// Synthetic sensor: given a true sensor->world orientation and a
	// true body-frame rate, produce what the IMU would report
	// (accelerometer reads specific force = +1g along "up")
	auto simulateAccel= [&](const glm::quat& sensorToWorld) {
		return glm::transpose(glm::mat3_cast(sensorToWorld)) * worldUp * kGravity;
	};

	// (a) Static, gravity only. A stationary sensor with a real
	// Joy-Con bias must hold level and learn the bias components it
	// CAN observe. Gravity constrains only 2 DoF (tilt), so the bias
	// about the gravity axis is unobservable here - the same physics
	// that makes yaw unobservable. Asserting that explicitly keeps
	// the limitation documented instead of surprising us later.
	{
		const glm::vec3 trueBias(0.007f, -0.033f, -0.006f); // measured Joy-Con R
		const glm::quat trueOrientation(1.f, 0.f, 0.f, 0.f); // level: gravity axis = sensor Z

		ImuOrientationFilter filter;
		filter.configure(ImuOrientationFilterConfig());

		const glm::vec3 accel= simulateAccel(trueOrientation);
		for (int step= 0; step < 200 * 30; ++step) // 30 seconds
		{
			ImuSample sample;
			sample.angularVelocity= trueBias; // stationary: the reading IS the bias
			sample.acceleration= accel;
			filter.processSample(sample, kDt);
		}

		const glm::vec3 estimatedBias= filter.getGyroBias();
		const glm::vec2 tiltBiasError(estimatedBias.x - trueBias.x, estimatedBias.y - trueBias.y);
		const float tiltBiasErrorMagnitude= glm::length(tiltBiasError);
		const float gravityAxisBiasError= fabsf(estimatedBias.z - trueBias.z);
		const glm::vec3 estimatedUp= glm::mat3_cast(filter.getOrientation()) * glm::vec3(0, 0, 1);
		const float tiltErrorDegrees= glm::degrees(acosf(std::clamp(estimatedUp.z, -1.f, 1.f)));

		MIKAN_LOG_INFO("test-imufilter")
			<< "(a) static/gravity-only: tilt-axis bias err=" << tiltBiasErrorMagnitude
			<< " rad/s, gravity-axis bias err=" << gravityAxisBiasError
			<< " rad/s (expected ~unobservable), tilt err=" << tiltErrorDegrees << " deg";
		if (tiltBiasErrorMagnitude > 0.002f || tiltErrorDegrees > 0.5f || !filter.isTiltConverged())
		{
			MIKAN_LOG_ERROR("test-imufilter")
				<< "(a) FAILED: must learn the OBSERVABLE bias axes and hold level";
			result= 1;
		}
		if (gravityAxisBiasError < 0.5f * fabsf(trueBias.z))
		{
			MIKAN_LOG_ERROR("test-imufilter")
				<< "(a) FAILED: gravity-axis bias appears observable - the gravity Jacobian is wrong";
			result= 1;
		}
	}

	// (a2) Same scenario WITH vision. Absolute orientation makes the
	// third bias axis observable, so the full 3-axis bias converges -
	// this is the concrete payoff of anchoring the filter to vision.
	{
		const glm::vec3 trueBias(0.007f, -0.033f, -0.006f);
		const glm::quat trueOrientation(1.f, 0.f, 0.f, 0.f);

		ImuOrientationFilter filter;
		filter.configure(ImuOrientationFilterConfig());

		const glm::vec3 accel= simulateAccel(trueOrientation);
		for (int step= 0; step < 200 * 30; ++step)
		{
			ImuSample sample;
			sample.angularVelocity= trueBias;
			sample.acceleration= accel;
			filter.processSample(sample, kDt);

			// Vision at ~30 Hz, as the tracker would supply it
			if (step % 7 == 0)
				filter.updateWithOrientation(trueOrientation);
		}

		const float biasError= glm::length(filter.getGyroBias() - trueBias);
		MIKAN_LOG_INFO("test-imufilter")
			<< "(a2) static/vision-aided: full bias err=" << biasError << " rad/s ("
			<< glm::degrees(biasError) << " deg/s)";
		if (biasError > 0.002f)
		{
			MIKAN_LOG_ERROR("test-imufilter")
				<< "(a2) FAILED: vision must make the full gyro bias observable";
			result= 1;
		}
	}

	// (b) Rotation tracking: sustained rotation about a tilted axis
	// with a bias present. Orientation must track the truth.
	{
		const glm::vec3 trueBias(0.01f, -0.02f, 0.005f);
		const glm::vec3 trueRate= glm::normalize(glm::vec3(0.3f, 0.6f, 0.2f)) * 1.2f; // rad/s

		ImuOrientationFilter filter;
		filter.configure(ImuOrientationFilterConfig());

		// Let it settle at rest first so the bias is known
		glm::quat trueOrientation(1.f, 0.f, 0.f, 0.f);
		for (int step= 0; step < 200 * 20; ++step)
		{
			ImuSample sample;
			sample.angularVelocity= trueBias;
			sample.acceleration= simulateAccel(trueOrientation);
			filter.processSample(sample, kDt);
		}

		// Now rotate for 4 seconds
		for (int step= 0; step < 200 * 4; ++step)
		{
			const float angle= glm::length(trueRate) * kDt;
			trueOrientation= glm::normalize(
				trueOrientation * glm::angleAxis(angle, glm::normalize(trueRate)));

			ImuSample sample;
			sample.angularVelocity= trueRate + trueBias;
			// Rotating in place: the accelerometer still sees only
			// gravity, so the gravity update stays valid
			sample.acceleration= simulateAccel(trueOrientation);
			filter.processSample(sample, kDt);
		}

		const glm::quat error= glm::inverse(trueOrientation) * filter.getOrientation();
		const float errorDegrees= glm::degrees(2.f * asinf(std::clamp(
			glm::length(glm::vec3(error.x, error.y, error.z)), 0.f, 1.f)));
		MIKAN_LOG_INFO("test-imufilter") << "(b) rotation: orientation err=" << errorDegrees << " deg";
		if (errorDegrees > 2.f)
		{
			MIKAN_LOG_ERROR("test-imufilter") << "(b) FAILED: must track sustained rotation";
			result= 1;
		}
	}

	// (c) Motion gating: a hard linear acceleration must NOT be
	// mistaken for gravity and tilt the estimate
	{
		ImuOrientationFilter filter;
		filter.configure(ImuOrientationFilterConfig());

		const glm::quat trueOrientation(1.f, 0.f, 0.f, 0.f);
		for (int step= 0; step < 200 * 10; ++step)
		{
			ImuSample sample;
			sample.acceleration= simulateAccel(trueOrientation);
			filter.processSample(sample, kDt);
		}
		const glm::vec3 upBefore= glm::mat3_cast(filter.getOrientation()) * glm::vec3(0, 0, 1);

		// 5 m/s^2 sideways shove for half a second
		int rejectedCount= 0;
		for (int step= 0; step < 100; ++step)
		{
			filter.predict(glm::vec3(0.f), kDt);
			const glm::vec3 shoved= simulateAccel(trueOrientation) + glm::vec3(5.f, 0.f, 0.f);
			if (!filter.updateWithGravity(shoved))
				rejectedCount++;
		}
		const glm::vec3 upAfter= glm::mat3_cast(filter.getOrientation()) * glm::vec3(0, 0, 1);
		const float tiltChangeDegrees=
			glm::degrees(acosf(std::clamp(glm::dot(upBefore, upAfter), -1.f, 1.f)));

		MIKAN_LOG_INFO("test-imufilter") << "(c) gating: rejected " << rejectedCount
			<< "/100 accelerated samples, tilt moved " << tiltChangeDegrees << " deg";
		if (rejectedCount != 100 || tiltChangeDegrees > 0.1f)
		{
			MIKAN_LOG_ERROR("test-imufilter")
				<< "(c) FAILED: accelerated samples must be gated out of the gravity update";
			result= 1;
		}
	}

	// (d) Yaw is unobservable from inertial data alone, and vision
	// must be able to fix it. Sanity-checks the whole reason the
	// filter is vision-anchored.
	{
		ImuOrientationFilter filter;
		filter.configure(ImuOrientationFilterConfig());

		const glm::quat trueOrientation(1.f, 0.f, 0.f, 0.f);
		for (int step= 0; step < 200 * 10; ++step)
		{
			ImuSample sample;
			sample.acceleration= simulateAccel(trueOrientation);
			filter.processSample(sample, kDt);
		}

		const glm::vec3 sigmaBefore= filter.getOrientationSigma();

		// Vision says the sensor is yawed 40 degrees
		const glm::quat visionOrientation=
			glm::angleAxis(glm::radians(40.f), glm::vec3(0.f, 0.f, 1.f));
		for (int update= 0; update < 20; ++update)
			filter.updateWithOrientation(visionOrientation);

		const glm::vec3 sigmaAfter= filter.getOrientationSigma();
		const glm::quat error= glm::inverse(visionOrientation) * filter.getOrientation();
		const float errorDegrees= glm::degrees(2.f * asinf(std::clamp(
			glm::length(glm::vec3(error.x, error.y, error.z)), 0.f, 1.f)));

		MIKAN_LOG_INFO("test-imufilter")
			<< "(d) yaw: sigma z before=" << sigmaBefore.z << " after=" << sigmaAfter.z
			<< " rad, orientation err after vision=" << errorDegrees << " deg";
		// Before vision, yaw uncertainty must still be large (gravity
		// can't see it); after vision it must be small and correct
		if (sigmaBefore.z < 0.5f || sigmaAfter.z > 0.1f || errorDegrees > 3.f)
		{
			MIKAN_LOG_ERROR("test-imufilter")
				<< "(d) FAILED: yaw must be unobservable inertially and fixable by vision";
			result= 1;
		}
	}

	// (e) Mounting calibration round trip. The whole chain has three
	// places a transpose/inverse could hide (mounting capture,
	// forearm reconstruction, wrist rotation) and a mistake in any
	// one produces plausible-looking but wrong wrist angles, so
	// verify it end to end against a known truth.
	{
		// Arbitrary "how the strap happens to sit" rotation
		const glm::quat trueForearmToSensor= glm::normalize(
			glm::angleAxis(glm::radians(63.f), glm::normalize(glm::vec3(0.3f, -0.8f, 0.5f))));

		// CALIBRATION: wrist straight, so forearm frame == palm frame
		const glm::quat forearmAtCapture=
			glm::angleAxis(glm::radians(21.f), glm::vec3(0.f, 0.f, 1.f));
		const glm::quat palmAtCapture= forearmAtCapture; // straight wrist
		// The sensor reports q_sw where q_fw = q_sw * q_fs
		const glm::quat sensorAtCapture= forearmAtCapture * glm::inverse(trueForearmToSensor);

		// captureMounting computes inverse(q_sw) * q_palm
		const glm::quat capturedMounting=
			glm::normalize(glm::inverse(sensorAtCapture) * palmAtCapture);
		const glm::quat mountingError= glm::inverse(trueForearmToSensor) * capturedMounting;
		const float mountingErrorDegrees= glm::degrees(2.f * asinf(std::clamp(
			glm::length(glm::vec3(mountingError.x, mountingError.y, mountingError.z)), 0.f, 1.f)));

		// RUNTIME: forearm moved AND the wrist is now bent 35 deg
		const glm::quat forearmNow= glm::normalize(
			glm::angleAxis(glm::radians(-40.f), glm::normalize(glm::vec3(0.2f, 0.9f, 0.1f))));
		const glm::quat trueWristLocal=
			glm::angleAxis(glm::radians(35.f), glm::vec3(1.f, 0.f, 0.f));
		const glm::quat palmNow= forearmNow * trueWristLocal;
		const glm::quat sensorNow= forearmNow * glm::inverse(trueForearmToSensor);

		// getForearmOrientation computes q_sw * q_fs
		const glm::quat reconstructedForearm= glm::normalize(sensorNow * capturedMounting);
		const glm::quat forearmError= glm::inverse(forearmNow) * reconstructedForearm;
		const float forearmErrorDegrees= glm::degrees(2.f * asinf(std::clamp(
			glm::length(glm::vec3(forearmError.x, forearmError.y, forearmError.z)), 0.f, 1.f)));

		// HandPose::getWristRotation computes inverse(forearm) * palm
		HandPose pose;
		pose.hasWorldPose= true;
		pose.hasForearmPose= true;
		pose.palmOrientationWorld= palmNow;
		pose.forearmOrientationWorld= reconstructedForearm;
		const glm::quat wristError= glm::inverse(trueWristLocal) * pose.getWristRotation();
		const float wristErrorDegrees= glm::degrees(2.f * asinf(std::clamp(
			glm::length(glm::vec3(wristError.x, wristError.y, wristError.z)), 0.f, 1.f)));

		// And a straight wrist must read identity, whatever the mounting
		HandPose straightPose;
		straightPose.hasWorldPose= true;
		straightPose.hasForearmPose= true;
		straightPose.palmOrientationWorld= forearmNow;
		straightPose.forearmOrientationWorld= reconstructedForearm;
		const float straightDegrees= glm::degrees(2.f * asinf(std::clamp(
			glm::length(glm::vec3(straightPose.getWristRotation().x,
								  straightPose.getWristRotation().y,
								  straightPose.getWristRotation().z)), 0.f, 1.f)));

		MIKAN_LOG_INFO("test-imufilter")
			<< "(e) mounting round trip: mounting err=" << mountingErrorDegrees
			<< " deg, forearm err=" << forearmErrorDegrees << " deg, wrist err="
			<< wristErrorDegrees << " deg, straight-wrist reads " << straightDegrees << " deg";
		if (mountingErrorDegrees > 0.01f || forearmErrorDegrees > 0.01f ||
			wristErrorDegrees > 0.01f || straightDegrees > 0.01f)
		{
			MIKAN_LOG_ERROR("test-imufilter")
				<< "(e) FAILED: mounting calibration / wrist rotation chain is inconsistent";
			result= 1;
		}
	}

	// (f) Motion-based arm-axis recovery. A held pose alone has to get
	// all three mounting DoF right at one instant, and in practice got
	// the ARM AXIS wrong by ~60 deg - which is the one the elbow rides
	// on. Twisting the forearm measures that axis directly: pronation
	// rotates about the arm and nothing else, so the dominant axis of
	// the sensor-frame angular-velocity scatter IS the arm axis.
	{
		const glm::quat trueMounting= glm::normalize(
			glm::angleAxis(glm::radians(115.f), glm::normalize(glm::vec3(0.3f, -0.7f, 0.6f))));
		// Arm axis (forearm +X) as the SENSOR sees it
		const glm::vec3 trueSensorArmAxis=
			glm::normalize(trueMounting * glm::vec3(1.f, 0.f, 0.f));

		auto accumulate= [](glm::mat3& scatter, const glm::vec3& rate, float weight) {
			for (int col= 0; col < 3; ++col)
				for (int row= 0; row < 3; ++row)
					scatter[col][row]+= rate[col] * rate[row] * weight;
		};

		// Simulated twisting: mostly about the arm axis, with a little
		// off-axis wobble because no one twists perfectly
		glm::mat3 twistScatter(0.f);
		const glm::vec3 offAxis= glm::normalize(glm::cross(trueSensorArmAxis, glm::vec3(0.f, 0.f, 1.f)));
		for (int sampleIndex= 0; sampleIndex < 400; ++sampleIndex)
		{
			const float phase= (float)sampleIndex * 0.05f;
			const glm::vec3 rate=
				trueSensorArmAxis * (3.f * sinf(phase)) + offAxis * (0.3f * sinf(phase * 2.7f));
			accumulate(twistScatter, rate, 0.005f);
		}

		float twistDominance= 0.f;
		const glm::vec3 measuredAxis= imuDominantRotationAxis(twistScatter, twistDominance);
		const float axisErrorDegrees= glm::degrees(
			acosf(std::clamp(fabsf(glm::dot(measuredAxis, trueSensorArmAxis)), 0.f, 1.f)));

		// Isotropic motion (waving the arm around, no real twist) must
		// score as uninformative so the UI can refuse the capture
		glm::mat3 isotropicScatter(0.f);
		accumulate(isotropicScatter, glm::vec3(1.f, 0.f, 0.f), 1.f);
		accumulate(isotropicScatter, glm::vec3(0.f, 1.f, 0.f), 1.f);
		accumulate(isotropicScatter, glm::vec3(0.f, 0.f, 1.f), 1.f);
		float isotropicDominance= 0.f;
		imuDominantRotationAxis(isotropicScatter, isotropicDominance);

		MIKAN_LOG_INFO("test-imufilter")
			<< "(f) motion axis: measured axis err=" << axisErrorDegrees
			<< " deg (dominance=" << twistDominance
			<< "), isotropic dominance=" << isotropicDominance;
		if (axisErrorDegrees > 2.f || twistDominance < 0.9f || isotropicDominance > 0.4f)
		{
			MIKAN_LOG_ERROR("test-imufilter")
				<< "(f) FAILED: motion-based arm axis recovery is wrong";
			result= 1;
		}
	}

	// (n) TWO-MOTION MOUNTING SOLVE. A twist fixes the forearm's long
	// axis, a curl about the elbow hinge fixes the roll that a twist
	// cannot see, and the accelerometer's centripetal term says which
	// end of the axis is the hand. No vision enters the geometry at
	// all, which is the entire point: the vision-averaged mounting it
	// replaces was corrupted by a wrist that would not hold still, and
	// averaging harder could not fix that.
	{
		const glm::quat trueMounting= glm::normalize(
			glm::angleAxis(glm::radians(115.f), glm::normalize(glm::vec3(0.3f, -0.7f, 0.6f))));
		const glm::vec3 armAxisSensor= glm::normalize(trueMounting * glm::vec3(1.f, 0.f, 0.f));
		const glm::vec3 hingeAxisSensor= glm::normalize(trueMounting * glm::vec3(0.f, 1.f, 0.f));
		const glm::vec3 wobbleAxis=
			glm::normalize(glm::cross(armAxisSensor, glm::vec3(0.f, 0.f, 1.f)));
		constexpr float k_trueRadiusMeters= 0.22f;

		auto mountingErrorDegrees= [](const glm::quat& a, const glm::quat& b) {
			glm::quat delta= glm::inverse(a) * b;
			if (delta.w < 0.f) // double cover: measure the short way round
				delta= -delta;
			return glm::degrees(2.f * asinf(std::clamp(
				glm::length(glm::vec3(delta.x, delta.y, delta.z)), 0.f, 1.f)));
		};

		// TWIST: back-and-forth about the long axis, with the off-axis
		// wobble nobody actually avoids
		std::vector<MotionSample> twist;
		for (int sampleIndex= 0; sampleIndex < 1200; ++sampleIndex)
		{
			const float phase= (float)sampleIndex * 0.02f;
			MotionSample sample;
			sample.rate= armAxisSensor * (4.f * sinf(phase)) +
				wobbleAxis * (0.3f * sinf(phase * 2.7f));
			sample.acceleration= glm::vec3(0.f, 0.f, 9.81f);
			sample.dtSeconds= 0.005f;
			twist.push_back(sample);
		}

		// CURL: back-and-forth about the hinge, with the accelerometer
		// carrying what a real one would - gravity as seen from a
		// rotating frame, the centripetal term pointing at the elbow,
		// and the tangential term perpendicular to both
		auto buildCurl= [&](const glm::vec3& axis) {
			std::vector<MotionSample> curl;
			glm::vec3 up= glm::normalize(glm::vec3(0.2f, 0.3f, 1.f));
			float previousRate= 0.f;
			for (int sampleIndex= 0; sampleIndex < 1600; ++sampleIndex)
			{
				constexpr float dt= 0.005f;
				const float phase= (float)sampleIndex * 0.02f;
				const float hingeRate= 3.5f * sinf(phase);
				const glm::vec3 rate=
					axis * hingeRate + wobbleAxis * (0.15f * sinf(phase * 3.3f));

				up= glm::normalize(up - glm::cross(rate, up) * dt);
				const float angularAcceleration= (hingeRate - previousRate) / dt;
				previousRate= hingeRate;

				MotionSample sample;
				sample.rate= rate;
				sample.acceleration= up * 9.81f -
					armAxisSensor * (hingeRate * hingeRate * k_trueRadiusMeters) +
					glm::cross(axis * angularAcceleration, armAxisSensor * k_trueRadiusMeters);
				sample.dtSeconds= dt;
				curl.push_back(sample);
			}
			return curl;
		};
		const std::vector<MotionSample> curl= buildCurl(hingeAxisSensor);

		// A deliberately TERRIBLE palmar hint: 50 degrees off, which is
		// worse than the 54 degree pose spread that made the old solve
		// unusable. It only has to choose between two candidates a half
		// turn apart, so it must still land on the right one.
		const glm::quat badHint= glm::normalize(
			glm::angleAxis(glm::radians(50.f), glm::normalize(glm::vec3(0.5f, 0.6f, -0.6f))) *
			trueMounting);

		MountingCaptureResult solved;
		imuSolveMountingFromMotions(twist, curl, &badHint, ePalmarSource::Vision, solved);
		const float solvedErrorDegrees= mountingErrorDegrees(trueMounting, solved.forearmToSensor);

		// The hint decides the palmar bit and nothing else, so a hint on
		// the OTHER side must move the answer by exactly a half turn -
		// proving the bit is driven by the reference rather than baked in
		const glm::quat flippedTruth= glm::normalize(
			trueMounting * glm::angleAxis(glm::radians(180.f), glm::vec3(1.f, 0.f, 0.f)));
		MountingCaptureResult flippedSolved;
		imuSolveMountingFromMotions(twist, curl, &flippedTruth, ePalmarSource::Vision, flippedSolved);
		const float flippedErrorDegrees=
			mountingErrorDegrees(flippedTruth, flippedSolved.forearmToSensor);

		// A curl that turns about the SAME axis as the twist (the
		// shoulder rotating instead of the elbow bending) leaves roll
		// unobservable, so it has to be refused rather than extrapolated
		MountingCaptureResult degenerate;
		imuSolveMountingFromMotions(twist, buildCurl(armAxisSensor), &badHint,
									ePalmarSource::Vision, degenerate);

		// No reference at all: the palmar side is genuinely unknown, and
		// guessing it is a coin flip that silently mirrors the hand
		MountingCaptureResult unhinted;
		imuSolveMountingFromMotions(twist, curl, nullptr, ePalmarSource::None, unhinted);

		MIKAN_LOG_INFO("test-imufilter")
			<< "(n) two-motion solve: err=" << solvedErrorDegrees << " deg, inter-axis="
			<< solved.interAxisAngleDegrees << " deg, hinge spread=" << solved.hingeSpreadDegrees
			<< " deg over " << solved.curlStrokes << " strokes, radius="
			<< solved.forearmLengthMeters << " m (r=" << solved.lengthFitCorrelation
			<< "), flipped-hint err=" << flippedErrorDegrees
			<< " deg, parallel-curl usable=" << degenerate.bMotionUsable
			<< " (inter-axis " << degenerate.interAxisAngleDegrees
			<< " deg), unhinted usable=" << unhinted.bMotionUsable;

		if (!solved.bMotionUsable || solvedErrorDegrees > 3.f ||
			solved.interAxisAngleDegrees < 85.f || solved.hingeSpreadDegrees > 5.f ||
			solved.curlStrokes < 3 || !solved.bLengthMeasured ||
			fabsf(solved.forearmLengthMeters - k_trueRadiusMeters) > 0.02f ||
			flippedErrorDegrees > 3.f || degenerate.bMotionUsable ||
			degenerate.interAxisAngleDegrees > 20.f || unhinted.bMotionUsable ||
			unhinted.palmarSource != ePalmarSource::None)
		{
			MIKAN_LOG_ERROR("test-imufilter")
				<< "(n) FAILED: two-motion mounting solve is wrong";
			result= 1;
		}
	}

	// (g) Twist gating. Both live failures came down to the same thing:
	// dominance alone accepts anything. A scatter is rank-1 - so
	// dominance ~1.0 - for a moment's motion, and ALSO for a constant
	// rate offset. A Joy-Con whose bias had run away to -5371 deg/s
	// scored 0.9999 while sitting on a desk, and the wizard's twist
	// stage completed instantly.
	{
		auto accumulate= [](glm::mat3& scatter, float& path, glm::vec3& net, const glm::vec3& rate,
							float dt) {
			const float magnitude= glm::length(rate);
			for (int col= 0; col < 3; ++col)
				for (int row= 0; row < 3; ++row)
					scatter[col][row]+= rate[col] * rate[row] * dt;
			path+= magnitude * dt;
			net+= rate * dt;
		};

		const glm::vec3 armAxis= glm::normalize(glm::vec3(0.4f, -0.8f, 0.45f));

		// A brief flick: single-axis by construction, but nowhere near
		// enough rotation to locate anything
		glm::mat3 flickScatter(0.f);
		float flickPath= 0.f;
		glm::vec3 flickNet(0.f);
		for (int i= 0; i < 20; ++i)
			accumulate(flickScatter, flickPath, flickNet, armAxis * 0.6f, 0.005f);

		// A constant rate - what a runaway gyro bias looks like. Lots of
		// "rotation", perfectly single-axis, and completely meaningless.
		glm::mat3 driftScatter(0.f);
		float driftPath= 0.f;
		glm::vec3 driftNet(0.f);
		for (int i= 0; i < 2000; ++i)
			accumulate(driftScatter, driftPath, driftNet, armAxis * 94.f, 0.005f);

		// Real pronation/supination: back and forth about the arm axis
		glm::mat3 twistScatter(0.f);
		float twistPath= 0.f;
		glm::vec3 twistNet(0.f);
		const glm::vec3 wobbleAxis= glm::normalize(glm::cross(armAxis, glm::vec3(0.f, 0.f, 1.f)));
		for (int i= 0; i < 800; ++i)
		{
			const float phase= (float)i * 0.05f;
			accumulate(twistScatter, twistPath, twistNet,
					   armAxis * (3.f * sinf(phase)) + wobbleAxis * (0.25f * sinf(phase * 2.7f)),
					   0.005f);
		}

		struct TwistCase
		{
			const char* name;
			const glm::mat3* scatter;
			float path;
			const glm::vec3* net;
			bool bExpectUsable;
		};
		const TwistCase cases[]= {
			{"flick", &flickScatter, flickPath, &flickNet, false},
			{"constant-rate", &driftScatter, driftPath, &driftNet, false},
			{"back-and-forth", &twistScatter, twistPath, &twistNet, true},
		};

		for (const TwistCase& twistCase : cases)
		{
			float dominance= 0.f, progress= 0.f, reversal= 0.f;
			imuEvaluateTwist(*twistCase.scatter, twistCase.path, *twistCase.net, dominance, progress,
							 reversal);
			const bool bUsable= imuIsTwistUsable(dominance, progress, reversal);
			MIKAN_LOG_INFO("test-imufilter")
				<< "(g) twist " << twistCase.name << ": dominance=" << dominance
				<< " progress=" << progress << " reversal=" << reversal << " usable=" << bUsable;
			if (bUsable != twistCase.bExpectUsable)
			{
				MIKAN_LOG_ERROR("test-imufilter")
					<< "(g) FAILED: '" << twistCase.name << "' should"
					<< (twistCase.bExpectUsable ? " " : " NOT ") << "be usable";
				result= 1;
			}
		}

		// The two rejected cases must still LOOK single-axis - that is
		// the whole point of the extra gates
		float dominance= 0.f, progress= 0.f, reversal= 0.f;
		imuEvaluateTwist(driftScatter, driftPath, driftNet, dominance, progress, reversal);
		if (dominance < 0.99f || reversal > 0.01f)
		{
			MIKAN_LOG_ERROR("test-imufilter")
				<< "(g) FAILED: a constant rate should score dominance ~1 and reversal ~0, got "
				<< dominance << " / " << reversal;
			result= 1;
		}
	}

	// (h) Gyro bias bound. The bias feeds back into predict(), so an
	// unbounded estimate is self-amplifying rather than self-correcting.
	{
		ImuOrientationFilter filter;
		ImuOrientationFilterConfig config;
		filter.configure(config);
		filter.initializeFromGravity(glm::vec3(0.f, 0.f, 9.80665f));

		filter.setGyroBias(glm::vec3(94.f, -50.f, 3.f)); // absurd, as observed live
		const glm::vec3 bounded= filter.getGyroBias();
		const bool bClamped= fabsf(bounded.x - config.maxGyroBias) < 1e-5f &&
							 fabsf(bounded.y + config.maxGyroBias) < 1e-5f &&
							 fabsf(bounded.z - config.maxGyroBias) < 1e-5f;

		// A sane measured bias must survive untouched
		filter.setGyroBias(glm::vec3(0.01f, -0.02f, 0.005f));
		const glm::vec3 kept= filter.getGyroBias();
		const bool bKept= fabsf(kept.x - 0.01f) < 1e-6f && fabsf(kept.y + 0.02f) < 1e-6f &&
						  fabsf(kept.z - 0.005f) < 1e-6f;

		MIKAN_LOG_INFO("test-imufilter")
			<< "(h) bias bound: clamped to " << bounded.x << ", " << bounded.y << ", " << bounded.z
			<< " rad/s; measured bias preserved=" << bKept;
		if (!bClamped || !bKept)
		{
			MIKAN_LOG_ERROR("test-imufilter") << "(h) FAILED: gyro bias bound is not enforced";
			result= 1;
		}
	}

	// (i) Wrist axial residual as a mounting-roll health check. The
	// wrist has no axial degree of freedom, so any axial rotation in
	// the measured joint is mounting roll error. Nothing acts on this
	// automatically - the curl measures roll properly now, and an
	// earlier version that DID correct it was steering a good roll
	// with the noisiest signal in the system - but it still has to
	// SEE the error, or a bad roll has no symptom.
	{
		const glm::quat trueMounting= glm::normalize(
			glm::angleAxis(glm::radians(174.f), glm::normalize(glm::vec3(-0.14f, 0.94f, 0.31f))));
		const glm::quat rollError= glm::angleAxis(glm::radians(45.f), glm::vec3(1.f, 0.f, 0.f));
		const glm::quat badMounting= glm::normalize(trueMounting * rollError);

		auto axialDegrees= [](const glm::quat& rotation) {
			glm::quat q= rotation;
			if (q.w < 0.f)
				q= -q;
			const glm::vec3 axisPart(q.x, q.y, q.z);
			const float axisLength= glm::length(axisPart);
			if (axisLength < 1e-6f)
				return 0.f;
			const float angle= 2.f * asinf(std::min(1.f, axisLength));
			return glm::degrees(angle * (axisPart.x / axisLength));
		};

		// The arm moves and the wrist genuinely flexes, but never
		// twists - so a correct mounting must stay near zero while the
		// rolled one reads its error, both under real motion.
		//
		// "Near zero" rather than zero: composing flexion with
		// deviation produces a couple of degrees of genuine axial
		// component, which is the wrist's own slack and is why the UI
		// only calls this bad above 5 degrees.
		float worstGoodResidual= 0.f;
		float meanBadResidual= 0.f;
		constexpr int k_steps= 4000;
		for (int step= 0; step < k_steps; ++step)
		{
			const float t= (float)step * 0.01f;
			const glm::quat sensorToWorld= glm::normalize(
				glm::angleAxis(0.7f * sinf(t), glm::normalize(glm::vec3(0.2f, 0.3f, 0.93f))));
			// Real wrist motion: flexion and deviation only, no twist
			const glm::quat trueWrist=
				glm::normalize(glm::angleAxis(0.4f * sinf(t * 1.7f), glm::vec3(0.f, 1.f, 0.f)) *
							   glm::angleAxis(0.2f * sinf(t * 2.3f), glm::vec3(0.f, 0.f, 1.f)));
			const glm::quat palm= glm::normalize(sensorToWorld * trueMounting * trueWrist);

			const glm::quat goodForearm= glm::normalize(sensorToWorld * trueMounting);
			worstGoodResidual= std::max(worstGoodResidual,
				fabsf(axialDegrees(glm::normalize(glm::inverse(goodForearm) * palm))));

			const glm::quat badForearm= glm::normalize(sensorToWorld * badMounting);
			meanBadResidual+=
				fabsf(axialDegrees(glm::normalize(glm::inverse(badForearm) * palm))) / (float)k_steps;
		}

		MIKAN_LOG_INFO("test-imufilter")
			<< "(i) axial residual: correct mounting reads at most " << worstGoodResidual
			<< " deg, a 45 deg rolled one averages " << meanBadResidual << " deg";

		if (worstGoodResidual > 3.f || meanBadResidual < 40.f)
		{
			MIKAN_LOG_ERROR("test-imufilter")
				<< "(i) FAILED: the wrist axial residual does not report mounting roll error";
			result= 1;
		}
	}

	// (k) Averaging the pose mounting. inverse(q_sensor) * q_palm is
	// CONSTANT under the correct model - both terms rotate with the
	// arm - so wrist bend is the only thing perturbing it, and the
	// mean recovers the mounting. This is what replaced sampling one
	// held pose, where a wrist 90+ deg off flipped the arm-axis
	// eigenvector and locked in a reproducible 180 deg error.
	{
		const glm::quat trueMounting= glm::normalize(
			glm::angleAxis(glm::radians(167.f), glm::normalize(glm::vec3(-0.02f, -0.09f, 0.99f))));

		glm::vec4 sum(0.f);
		int samples= 0;
		float worstSingleSampleDegrees= 0.f;
		for (int step= 0; step < 600; ++step)
		{
			const float t= (float)step * 0.02f;
			// The arm sweeps through many orientations as it twists
			const glm::quat sensorToWorld= glm::normalize(
				glm::angleAxis(2.2f * sinf(t), glm::normalize(glm::vec3(0.1f, -0.2f, 0.97f))) *
				glm::angleAxis(0.5f * sinf(t * 0.7f), glm::vec3(1.f, 0.f, 0.f)));
			// Wrist bend: bounded, zero-mean, and large enough that a
			// single unlucky sample is badly wrong
			const glm::quat wristBend= glm::normalize(
				glm::angleAxis(0.9f * sinf(t * 1.9f), glm::vec3(0.f, 1.f, 0.f)) *
				glm::angleAxis(0.6f * sinf(t * 2.6f), glm::vec3(0.f, 0.f, 1.f)));
			const glm::quat palm= glm::normalize(sensorToWorld * trueMounting * wristBend);

			const glm::quat sample= glm::normalize(glm::inverse(sensorToWorld) * palm);
			const glm::quat singleError= glm::inverse(trueMounting) * sample;
			worstSingleSampleDegrees= std::max(worstSingleSampleDegrees,
				glm::degrees(2.f * asinf(std::clamp(
					glm::length(glm::vec3(singleError.x, singleError.y, singleError.z)), 0.f, 1.f))));

			// Same hemisphere alignment the service applies
			glm::quat aligned= sample;
			if (samples > 0)
			{
				const glm::quat mean= glm::normalize(glm::quat(sum.w, sum.x, sum.y, sum.z));
				if (glm::dot(aligned, mean) < 0.f)
					aligned= -aligned;
			}
			sum+= glm::vec4(aligned.x, aligned.y, aligned.z, aligned.w);
			samples++;
		}

		const glm::quat mean= glm::normalize(glm::quat(sum.w, sum.x, sum.y, sum.z));
		const glm::quat meanError= glm::inverse(trueMounting) * mean;
		const float meanErrorDegrees= glm::degrees(2.f * asinf(std::clamp(
			glm::length(glm::vec3(meanError.x, meanError.y, meanError.z)), 0.f, 1.f)));

		// The arm axis sign is resolved against the pose mounting, so
		// the mean's +X must agree with the truth's - that is the
		// specific failure this replaced
		const glm::vec3 trueAxis= glm::normalize(trueMounting * glm::vec3(1.f, 0.f, 0.f));
		const glm::vec3 meanAxis= glm::normalize(mean * glm::vec3(1.f, 0.f, 0.f));
		const float axisAgreement= glm::dot(trueAxis, meanAxis);

		MIKAN_LOG_INFO("test-imufilter")
			<< "(k) pose averaging: worst single sample " << worstSingleSampleDegrees
			<< " deg off, mean " << meanErrorDegrees << " deg off, arm-axis agreement "
			<< axisAgreement;

		// The point of the test: single samples are badly wrong, the
		// mean is not, and the sign never flips
		if (worstSingleSampleDegrees < 45.f || meanErrorDegrees > 15.f || axisAgreement < 0.9f)
		{
			MIKAN_LOG_ERROR("test-imufilter")
				<< "(k) FAILED: averaging did not recover the mounting from noisy samples";
			result= 1;
		}
	}

	// (l) Yaw covariance must RE-INFLATE. Yaw is not observable from
	// inertial data, so its uncertainty has to grow between vision
	// corrections. Without that it collapses on the first correction
	// and never recovers, and since the Kalman gain is P/(P+R) a tiny
	// prior against a loose vision measurement switches the anchor off
	// in all but name - measured live as yaw sigma pinned at 0.010 rad
	// with the applied correction reading 0.00 deg.
	{
		ImuOrientationFilter filter;
		ImuOrientationFilterConfig config;
		filter.configure(config);
		filter.initializeFromGravity(glm::vec3(0.f, 0.f, 9.80665f));

		// Settle, then let vision pin yaw hard
		for (int step= 0; step < 400; ++step)
		{
			filter.predict(glm::vec3(0.f), 1.f / 200.f);
			filter.updateWithGravity(glm::vec3(0.f, 0.f, 9.80665f));
		}
		for (int step= 0; step < 200; ++step)
			filter.updateWithYawReference(glm::quat(1.f, 0.f, 0.f, 0.f), 0.05f);

		const float pinnedYaw= filter.getOrientationSigma().z;

		// Now coast on the gyro alone for two seconds
		for (int step= 0; step < 400; ++step)
		{
			filter.predict(glm::vec3(0.f), 1.f / 200.f);
			filter.updateWithGravity(glm::vec3(0.f, 0.f, 9.80665f));
		}
		const float coastedYaw= filter.getOrientationSigma().z;
		const float coastedTilt= filter.getTiltSigma();

		// A vision correction must now actually move the estimate
		const glm::quat reference= glm::angleAxis(glm::radians(20.f), glm::vec3(0.f, 0.f, 1.f));
		const glm::quat before= filter.getOrientation();
		for (int step= 0; step < 30; ++step)
			filter.updateWithYawReference(reference, 0.31f);
		const glm::quat moved= glm::inverse(before) * filter.getOrientation();
		const float movedDegrees= glm::degrees(2.f * asinf(std::clamp(
			glm::length(glm::vec3(moved.x, moved.y, moved.z)), 0.f, 1.f)));

		MIKAN_LOG_INFO("test-imufilter")
			<< "(l) yaw covariance: pinned " << pinnedYaw << " rad, re-inflated to " << coastedYaw
			<< " after 2s coasting (tilt stayed " << coastedTilt << "), vision then moved yaw "
			<< movedDegrees << " deg";

		// Yaw must grow, tilt must NOT (gravity observes it), and the
		// anchor must have real authority again
		if (coastedYaw <= pinnedYaw * 1.5f || coastedTilt > 0.05f || movedDegrees < 1.f)
		{
			MIKAN_LOG_ERROR("test-imufilter")
				<< "(l) FAILED: yaw uncertainty is not recovering, so the vision anchor is inert";
			result= 1;
		}
	}

	// (m) Palmar side must be temporally stable.
	//
	// The sign is decided per frame from geometry. On a nearly flat
	// hand the evidence is weak and its SIGN follows landmark noise,
	// so the palm frame inverts between frames - measured live as the
	// left palm frame flipping twice in 240 frames while the right
	// never did, which poisoned the mounting average with two clusters
	// 180 deg apart (70 deg pose spread against the right's 13).
	//
	// The test asserts the CONTRAST: the unprotected path must flip,
	// or the fix is being credited for nothing.
	{
		// A flat hand whose fingers carry a barely-there curl. The
		// curl sign alternates, which is what landmark noise does to a
		// hand held flat, and it keeps the score inside the weak band
		// where the old code had no stable answer.
		auto buildHand= [](float curlSign) {
			std::array<glm::vec3, HAND_LANDMARK_COUNT> points{};
			points[(int)eHandLandmark::WRIST]= glm::vec3(0.f, 0.f, 0.f);
			const float spread[FINGER_COUNT]= {0.035f, 0.02f, 0.f, -0.018f, -0.032f};
			for (int finger= 0; finger < FINGER_COUNT; ++finger)
			{
				const int* joints= FINGER_JOINTS[finger];
				const glm::vec3 base(0.075f, spread[finger], 0.f);
				points[joints[0]]= base;
				for (int joint= 1; joint < 4; ++joint)
				{
					const float along= 0.022f * (float)joint;
					// Quadratic droop = a gentle curl; ~0.5 deg per joint
					const float curl= curlSign * 0.00018f * (float)(joint * joint);
					points[joints[joint]]= base + glm::vec3(along, 0.f, curl);
				}
			}
			return points;
		};

		auto runSequence= [&buildHand](HandPoseModel::PalmarSideMemory* memory) {
			int flips= 0;
			glm::vec3 previousZ(0.f);
			for (int step= 0; step < 120; ++step)
			{
				// Sign alternates every few frames, as noise would
				const float curlSign= sinf((float)step * 1.1f) > 0.f ? 1.f : -1.f;
				const std::array<glm::vec3, HAND_LANDMARK_COUNT> points= buildHand(curlSign);
				const glm::vec3 z=
					glm::vec3(HandPoseModel::computePalmFrame(points, eHandSide::Left, memory)[2]);
				if (step > 0 && glm::dot(z, previousZ) < 0.f)
					flips++;
				previousZ= z;
			}
			return flips;
		};

		HandPoseModel::PalmarSideMemory memory;
		const int flipsWithMemory= runSequence(&memory);
		const int flipsWithoutMemory= runSequence(nullptr);

		// A reacquired hand must not inherit the old side
		HandPoseModel::PalmarSideMemory cleared= memory;
		cleared.reset();
		const bool bClearedForgets= glm::dot(cleared.palmarNormal, cleared.palmarNormal) < 1e-6f;

		MIKAN_LOG_INFO("test-imufilter")
			<< "(m) palmar side: " << flipsWithMemory << " flips with memory, "
			<< flipsWithoutMemory << " without, reset clears=" << bClearedForgets;

		// flipsWithoutMemory > 0 is the teeth: without it this test
		// passes whether or not the memory does anything at all
		if (flipsWithoutMemory == 0)
		{
			MIKAN_LOG_ERROR("test-imufilter")
				<< "(m) FAILED: the unprotected path did not flip, so this proves nothing";
			result= 1;
		}
		else if (flipsWithMemory != 0 || !bClearedForgets)
		{
			MIKAN_LOG_ERROR("test-imufilter")
				<< "(m) FAILED: the palmar side is not temporally stable";
			result= 1;
		}
	}

	if (result == 0)
		MIKAN_LOG_INFO("test-imufilter") << "All IMU filter checks passed";

	return result;
}

MIKAN_REGISTER_TEST("--test-imufilter", "IMU orientation EKF, observability, gating", eTestCategory::SelfTest, runImuFilterTest);

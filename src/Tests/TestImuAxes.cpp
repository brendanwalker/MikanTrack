#include "TestCommon.h"

static int runImuAxesTest(const TestArgs& args)
{
	// Are the gyro axes consistent with the accelerometer axes?
	//
	// A mounting calibration can absorb any fixed ROTATION between the
	// sensor and the body it rides, but it cannot absorb an axis
	// PERMUTATION or sign flip between the two sensors inside the
	// chip - that isn't a rotation, and it makes integrated motion go
	// the wrong way while static tilt still looks fine.
	//
	// The check needs no integration and no ground truth. "Up" is
	// fixed in the world, so its direction in the SENSOR frame must
	// obey dg/dt = -omega x g. Score every signed permutation of the
	// gyro axes against the gravity motion the accelerometer actually
	// measured, and the right one wins outright.
	JoyconDeviceManager manager;
	manager.startup();
	const size_t deviceCount= manager.getDeviceCount();
	if (deviceCount == 0)
	{
		MIKAN_LOG_ERROR("test-imuaxes") << "No Joy-Cons paired";
		return 1;
	}
	for (size_t deviceIndex= 0; deviceIndex < deviceCount; ++deviceIndex)
		manager.getDeviceByIndex(deviceIndex)->open();

	// Collection time is a knob because it is the thing that decides
	// whether the result is conclusive: it takes a lot of varied,
	// slow rotation to accumulate enough usable windows. 12s was not
	// enough in practice; 30s was.
	int collectSeconds= 30;
	for (const std::string& argument : args)
	{
		const int parsed= atoi(argument.c_str());
		if (parsed > 0)
		{
			collectSeconds= parsed;
			break;
		}
	}

	MIKAN_LOG_INFO("test-imuaxes")
		<< "Collecting " << collectSeconds
		<< " seconds - SLOWLY rotate each controller through all three axes "
		   "(roll, pitch, yaw) in large sweeps, avoiding sharp shakes.";

	std::vector<std::vector<ImuSample>> collected(deviceCount);
	for (int second= 0; second < collectSeconds; ++second)
	{
		std::this_thread::sleep_for(std::chrono::seconds(1));
		for (size_t deviceIndex= 0; deviceIndex < deviceCount; ++deviceIndex)
			manager.getDeviceByIndex(deviceIndex)->fetchSamples(collected[deviceIndex]);
		MIKAN_LOG_INFO("test-imuaxes") << "  " << (second + 1) << "/" << collectSeconds;
	}

	// All 48 signed permutations: which axis of the raw gyro feeds
	// each output axis, and with what sign
	static const int kPermutations[6][3]= {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
										   {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
	const char* kAxisNames[3]= {"X", "Y", "Z"};

	int result= 0;
	for (size_t deviceIndex= 0; deviceIndex < deviceCount; ++deviceIndex)
	{
		std::shared_ptr<IImuDevice> device= manager.getDeviceByIndex(deviceIndex);
		const std::vector<ImuSample>& samples= collected[deviceIndex];
		if (samples.size() < 200)
		{
			MIKAN_LOG_ERROR("test-imuaxes")
				<< device->getFriendlyName() << ": only " << samples.size() << " samples";
			result= 1;
			continue;
		}

		// Gyro bias from the quietest second (so a resting stretch
		// anywhere in the capture serves as the zero reference)
		glm::vec3 bias(0.f);
		{
			float quietest= 1e9f;
			for (size_t start= 0; start + 200 < samples.size(); start+= 200)
			{
				glm::vec3 sum(0.f);
				float motion= 0.f;
				for (size_t k= start; k < start + 200; ++k)
				{
					sum+= samples[k].angularVelocity;
					motion+= glm::length(samples[k].angularVelocity);
				}
				if (motion < quietest)
				{
					quietest= motion;
					bias= sum / 200.f;
				}
			}
		}

		// Score over WINDOWS, not consecutive samples. Between two
		// samples 5ms apart the gyro term is |w|*dt ~ 0.005, far below
		// accelerometer noise, so every candidate scores the same and
		// the winner is noise. Integrating across ~0.4s of real
		// rotation makes the gyro term ~1 rad - orders of magnitude
		// above the noise floor - so a wrong mapping cannot hide.
		constexpr float kWindowSeconds= 0.4f;
		constexpr float kMinWindowRotationRadians= 0.35f; // ~20 deg: below this a window says nothing

		struct Window
		{
			size_t startIndex= 0;
			size_t endIndex= 0;
			glm::vec3 gravityStart{0.f};
			glm::vec3 gravityEnd{0.f};
		};
		std::vector<Window> windows;
		{
			size_t startIndex= 0;
			for (size_t k= 1; k < samples.size(); ++k)
			{
				const double elapsedMs= samples[k].timestampMs - samples[startIndex].timestampMs;
				if (elapsedMs < kWindowSeconds * 1000.0)
					continue;

				// Both ends must be reading gravity alone, or the
				// direction we are predicting isn't gravity
				const float startMagnitude= glm::length(samples[startIndex].acceleration);
				const float endMagnitude= glm::length(samples[k].acceleration);
				const bool bEndsAreGravity= fabsf(startMagnitude - 9.80665f) < 0.6f &&
					fabsf(endMagnitude - 9.80665f) < 0.6f;

				if (bEndsAreGravity)
				{
					Window window;
					window.startIndex= startIndex;
					window.endIndex= k;
					window.gravityStart= samples[startIndex].acceleration / startMagnitude;
					window.gravityEnd= samples[k].acceleration / endMagnitude;
					// Only keep windows containing real rotation
					if (glm::length(window.gravityEnd - window.gravityStart) > 0.15f)
						windows.push_back(window);
				}
				startIndex= k;
			}
		}

		if (windows.size() < 5)
		{
			MIKAN_LOG_ERROR("test-imuaxes")
				<< device->getFriendlyName() << ": only " << windows.size()
				<< " usable rotation windows - rotate the controller more (and more slowly)";
			result= 1;
			continue;
		}

		float bestScore= 1e30f, identityScore= 0.f, runnerUpScore= 1e30f;
		int bestPermutation= 0, bestSigns= 0;
		for (int permutationIndex= 0; permutationIndex < 6; ++permutationIndex)
		{
			for (int signMask= 0; signMask < 8; ++signMask)
			{
				const glm::vec3 signs((signMask & 1) ? -1.f : 1.f, (signMask & 2) ? -1.f : 1.f,
									  (signMask & 4) ? -1.f : 1.f);

				float score= 0.f;
				int usedWindows= 0;
				for (const Window& window : windows)
				{
					// Accumulate the body-frame rotation across the window
					glm::quat deltaRotation(1.f, 0.f, 0.f, 0.f);
					float sweptRadians= 0.f;
					for (size_t k= window.startIndex; k < window.endIndex; ++k)
					{
						const float dt=
							(float)((samples[k + 1].timestampMs - samples[k].timestampMs) / 1000.0);
						if (dt <= 0.f || dt > 0.05f)
							continue;

						const glm::vec3 rawRate= samples[k].angularVelocity - bias;
						const glm::vec3 mappedRate(
							signs.x * rawRate[kPermutations[permutationIndex][0]],
							signs.y * rawRate[kPermutations[permutationIndex][1]],
							signs.z * rawRate[kPermutations[permutationIndex][2]]);

						const float rate= glm::length(mappedRate);
						if (rate > 1e-9f)
						{
							deltaRotation= glm::normalize(
								deltaRotation * glm::angleAxis(rate * dt, mappedRate / rate));
							sweptRadians+= rate * dt;
						}
					}
					if (sweptRadians < kMinWindowRotationRadians)
						continue;

					// A world-fixed direction in body coordinates
					// transforms by the INVERSE of the body rotation
					const glm::vec3 predictedGravityEnd=
						glm::inverse(deltaRotation) * window.gravityStart;
					score+= glm::length(predictedGravityEnd - window.gravityEnd);
					usedWindows++;
				}

				if (usedWindows < 5)
					continue;
				score/= (float)usedWindows;

				const bool bIsIdentity= permutationIndex == 0 && signMask == 0;
				if (bIsIdentity)
					identityScore= score;
				if (score < bestScore)
				{
					runnerUpScore= bestScore;
					bestScore= score;
					bestPermutation= permutationIndex;
					bestSigns= signMask;
				}
				else if (score < runnerUpScore)
				{
					runnerUpScore= score;
				}
			}
		}

		char mapping[64];
		snprintf(mapping, sizeof(mapping), "(%s%s, %s%s, %s%s)",
				 (bestSigns & 1) ? "-" : "+", kAxisNames[kPermutations[bestPermutation][0]],
				 (bestSigns & 2) ? "-" : "+", kAxisNames[kPermutations[bestPermutation][1]],
				 (bestSigns & 4) ? "-" : "+", kAxisNames[kPermutations[bestPermutation][2]]);
		const bool bIsIdentity= bestPermutation == 0 && bestSigns == 0;

		// A candidate only means something if it beats the
		// alternatives DECISIVELY. A near-tie means the measurement
		// carried no information about the mapping (too little
		// rotation, or too much accelerometer noise), and reporting a
		// winner then would be reporting noise.
		constexpr float kDecisiveRatio= 2.f;
		const float identityRatio= bestScore > 1e-9f ? identityScore / bestScore : 1.f;
		const float runnerUpRatio= bestScore > 1e-9f ? runnerUpScore / bestScore : 1.f;

		MIKAN_LOG_INFO("test-imuaxes")
			<< device->getFriendlyName() << ": " << windows.size() << " windows | best " << mapping
			<< " residual " << bestScore << " | identity residual " << identityScore << " (x"
			<< identityRatio << ") | runner-up (x" << runnerUpRatio << ")";

		if (identityRatio < kDecisiveRatio && runnerUpRatio < kDecisiveRatio)
		{
			MIKAN_LOG_WARNING("test-imuaxes")
				<< "  -> INCONCLUSIVE: no mapping wins decisively (need >" << kDecisiveRatio
				<< "x). Rotate through larger, slower sweeps on all three axes and rerun.";
			result= 1;
		}
		else if (bIsIdentity)
		{
			MIKAN_LOG_INFO("test-imuaxes")
				<< "  -> gyro axes agree with the accelerometer (identity wins by x"
				<< runnerUpRatio << ")";
		}
		else
		{
			MIKAN_LOG_ERROR("test-imuaxes")
				<< "  -> MISMATCH: remap the gyro to " << mapping << " (beats identity by x"
				<< identityRatio << ")";
			result= 1;
		}
	}

	manager.shutdown();
	return result;
}

MIKAN_REGISTER_TEST("--test-imuaxes", "Live Joy-Con axis convention measurement", eTestCategory::Hardware, runImuAxesTest);

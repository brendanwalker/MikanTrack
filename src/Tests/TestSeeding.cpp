#include "TestCommon.h"

// Cross-camera seeding decides whether a projected hint duplicates a hand this
// camera already tracks. That decision used to be split between a LABEL check
// in the vision thread and a POSITIONAL check here, and after a hand leaves
// and re-enters the labels get displaced - so a camera holding one hand under
// the wrong label had one side skipped by the label check and the other
// suppressed by the positional one. Neither was seeded and reacquisition waited
// a full palm-detector interval. The label check is gone; this pins the
// positional one, which now carries the whole decision.
//
// The rest of the seeding path needs the ONNX models, so this covers the part
// that can be tested in isolation.

static int runSeedingTest(const TestArgs& args)
{
	int result= 0;

	auto check = [&result](bool bCondition, const char* label) {
		MIKAN_LOG_INFO("test-seeding") << (bCondition ? "  ok   " : "  FAIL ") << label;
		if (!bCondition)
			result= 1;
	};

	// A 100x80 hand box centred at (300, 200)
	const glm::vec2 boxMin(250.f, 160.f);
	const glm::vec2 boxMax(350.f, 240.f);
	const std::vector<HandBox> oneBox{{boxMin, boxMax}};

	// -- Nothing tracked yet: every hint is worth trying -----
	check(!HandTrackingPipeline::isSeedRedundant(glm::vec2(300.f, 200.f), {}),
		  "no active slots -> not redundant");

	// -- A hint on top of a tracked hand is the camera's own hand -----
	check(HandTrackingPipeline::isSeedRedundant(glm::vec2(300.f, 200.f), oneBox),
		  "hint at the box centre -> redundant");
	check(HandTrackingPipeline::isSeedRedundant(glm::vec2(255.f, 235.f), oneBox),
		  "hint inside the box -> redundant");

	// -- The projection-error margin -----
	// The hint carries the OTHER camera's depth error, which arrives here as a
	// lateral offset, so the box is inflated about its centre before the test.
	// At 1.25 the half-extents grow 50 -> 62.5 and 40 -> 50.
	check(HandTrackingPipeline::isSeedRedundant(glm::vec2(360.f, 200.f), oneBox),
		  "10 px outside the box, inside the margin -> redundant");
	check(HandTrackingPipeline::isSeedRedundant(glm::vec2(300.f, 248.f), oneBox),
		  "8 px below the box, inside the margin -> redundant");
	check(!HandTrackingPipeline::isSeedRedundant(glm::vec2(370.f, 200.f), oneBox),
		  "20 px outside the box, beyond the margin -> seed it");

	// -- A genuinely different hand still gets seeded -----
	check(!HandTrackingPipeline::isSeedRedundant(glm::vec2(700.f, 400.f), oneBox),
		  "far from every box -> seed it");

	// -- Inflation is about the box centre, not the origin -----
	// A box far from the origin would swallow the whole image if the scale
	// were applied to its corners instead.
	const std::vector<HandBox> farBox{{glm::vec2(1000.f, 600.f), glm::vec2(1100.f, 680.f)}};
	check(!HandTrackingPipeline::isSeedRedundant(glm::vec2(300.f, 200.f), farBox),
		  "distant box does not swallow the origin side of the frame");
	check(HandTrackingPipeline::isSeedRedundant(glm::vec2(1050.f, 640.f), farBox),
		  "distant box still matches its own centre");

	// -- Two tracked hands: either one can absorb a hint -----
	const std::vector<HandBox> twoBoxes{{boxMin, boxMax}, {glm::vec2(700.f, 400.f), glm::vec2(800.f, 480.f)}};
	check(HandTrackingPipeline::isSeedRedundant(glm::vec2(750.f, 440.f), twoBoxes),
		  "hint on the second box -> redundant");
	check(!HandTrackingPipeline::isSeedRedundant(glm::vec2(520.f, 320.f), twoBoxes),
		  "hint between the two boxes -> seed it");

	if (result == 0)
		MIKAN_LOG_INFO("test-seeding") << "All cross-camera seeding checks passed";

	return result;
}

MIKAN_REGISTER_TEST("--test-seeding",
					"Cross-camera seed redundancy gate",
					eTestCategory::SelfTest, runSeedingTest);

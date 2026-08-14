#include "TrackingJson.h"

using json= nlohmann::json;

namespace TrackingJson
{
json vec2ToJson(const glm::vec2& v)
{
	return json::array({v.x, v.y});
}

json vec3ToJson(const glm::vec3& v)
{
	return json::array({v.x, v.y, v.z});
}

json quatToJson(const glm::quat& q)
{
	return json::array({q.x, q.y, q.z, q.w}); // xyzw, matching the OSC schema
}

const char* sideName(int side)
{
	return side == 0 ? "left" : (side == 1 ? "right" : "none");
}

json fingersToJson(const std::array<FingerAngles, FINGER_COUNT>& fingers)
{
	json out= json::array();
	for (const FingerAngles& angles : fingers)
		out.push_back(json::array({angles.lateral, angles.proximal, angles.intermediate, angles.distal}));
	return out;
}

json landmarksToJson(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points)
{
	json out= json::array();
	for (const glm::vec3& p : points)
		out.push_back(vec3ToJson(p));
	return out;
}

json bodyPoseToJson(const BodyPoseObservation& body)
{
	if (!body.valid)
		return {{"valid", false}};

	json imagePoints= json::array();
	json visibility= json::array();
	json worldPoints= json::array();
	for (int landmark= 0; landmark < POSE_LANDMARK_COUNT; ++landmark)
	{
		imagePoints.push_back(vec3ToJson(body.imagePoints[landmark]));
		visibility.push_back(body.visibility[landmark]);
		worldPoints.push_back(vec3ToJson(body.worldPoints[landmark]));
	}
	return {
		{"valid", true},
		{"modelFrameIndex", body.modelFrameIndex},
		{"providedMask", body.providedMask},
		{"boxSource", (int)body.boxSource},
		{"confidence", body.confidence},
		{"imagePoints", imagePoints},
		{"visibility", visibility},
		{"worldPoints", worldPoints},
	};
}

json imageQualityToJson(const HandImageQuality& quality)
{
	return {
		{"meanLuma", quality.meanLuma},
		{"shadowClipRatio", quality.shadowClipRatio},
		{"highlightClipRatio", quality.highlightClipRatio},
		{"contrast", quality.contrast},
		{"backgroundSeparation", quality.backgroundSeparation},
		{"sharpness", quality.sharpness},
		{"noise", quality.noise},
	};
}

json fusionDiagnosticsToJson(const FusionDiagnostics& diagnostics)
{
	json clusters= json::array();
	for (const FusionDiagnostics::Cluster& cluster : diagnostics.clusters)
	{
		json observations= json::array();
		for (const FusionDiagnostics::Observation& observation : cluster.observations)
		{
			observations.push_back({
				{"camera", observation.cameraIndex},
				{"labeledSide", sideName(observation.labeledSide)},
				{"weight", observation.weight},
				{"confidence", observation.confidence},
				{"stability", observation.stability},
				{"jitterMm", observation.jitterMm},
				{"sideVoteWeight", observation.sideVoteWeight},
				{"palmWorld", vec3ToJson(observation.palmWorld)},
			});
		}

		json affinity;
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			affinity[sideName(sideIndex)]= {
				{"vote", cluster.affinity[sideIndex][0]},
				{"temporal", cluster.affinity[sideIndex][1]},
				{"spatial", cluster.affinity[sideIndex][2]},
				{"total",
				 cluster.affinity[sideIndex][0] + cluster.affinity[sideIndex][1] + cluster.affinity[sideIndex][2]},
			};
		}

		clusters.push_back({
			{"palmWorld", vec3ToJson(cluster.palmWorld)},
			{"bestWeight", cluster.bestWeight},
			{"assignedSide", sideName(cluster.assignedSide)},
			{"triangulated", cluster.triangulated},
			{"triVetoed", cluster.triVetoed},
			{"triResidualRmsPx", cluster.triResidualRmsPx},
			{"triResidualMaxPx", cluster.triResidualMaxPx},
			{"affinity", affinity},
			{"observations", observations},
		});
	}

	return {
		{"totalObservations", diagnostics.totalObservations},
		{"clusters", clusters},
	};
}

float floatFromJson(const json& value)
{
	return value.is_number() ? value.get<float>() : 0.f;
}

glm::vec2 vec2FromJson(const json& j)
{
	if (!j.is_array() || j.size() != 2)
		return glm::vec2(0.f);
	return glm::vec2(floatFromJson(j[0]), floatFromJson(j[1]));
}

glm::vec3 vec3FromJson(const json& j)
{
	if (!j.is_array() || j.size() != 3)
		return glm::vec3(0.f);
	return glm::vec3(floatFromJson(j[0]), floatFromJson(j[1]), floatFromJson(j[2]));
}

glm::quat quatFromJson(const json& j)
{
	if (!j.is_array() || j.size() != 4)
		return glm::quat(1.f, 0.f, 0.f, 0.f);
	// stored xyzw; glm constructor is (w, x, y, z)
	return glm::quat(floatFromJson(j[3]), floatFromJson(j[0]), floatFromJson(j[1]), floatFromJson(j[2]));
}

void fingersFromJson(const json& j, std::array<FingerAngles, FINGER_COUNT>& outFingers)
{
	outFingers= {};
	if (!j.is_array())
		return;
	for (int fingerIndex= 0; fingerIndex < FINGER_COUNT && fingerIndex < (int)j.size(); ++fingerIndex)
	{
		const json& angles= j[fingerIndex];
		if (!angles.is_array() || angles.size() != 4)
			continue;
		outFingers[fingerIndex].lateral= floatFromJson(angles[0]);
		outFingers[fingerIndex].proximal= floatFromJson(angles[1]);
		outFingers[fingerIndex].intermediate= floatFromJson(angles[2]);
		outFingers[fingerIndex].distal= floatFromJson(angles[3]);
	}
}

void landmarksFromJson(const json& j, std::array<glm::vec3, HAND_LANDMARK_COUNT>& outPoints)
{
	outPoints= {};
	if (!j.is_array())
		return;
	for (int landmark= 0; landmark < HAND_LANDMARK_COUNT && landmark < (int)j.size(); ++landmark)
		outPoints[landmark]= vec3FromJson(j[landmark]);
}

void bodyPoseFromJson(const json& j, BodyPoseObservation& outBody)
{
	outBody= BodyPoseObservation();
	if (!j.is_object() || !j.value("valid", false))
		return;
	outBody.valid= true;
	outBody.modelFrameIndex= j.value("modelFrameIndex", (int64_t)-1);
	// Recordings predating the mask carry the full 33-slot BlazePose layout
	outBody.providedMask= j.value("providedMask", (1u << POSE_LANDMARK_COUNT) - 1u);
	outBody.boxSource= (eBodyBoxSource)j.value("boxSource", (int)eBodyBoxSource::None);
	outBody.confidence= j.value("confidence", 0.f);
	const json& imagePoints= j.value("imagePoints", json::array());
	const json& visibility= j.value("visibility", json::array());
	const json& worldPoints= j.value("worldPoints", json::array());
	for (int landmark= 0; landmark < POSE_LANDMARK_COUNT; ++landmark)
	{
		if (landmark < (int)imagePoints.size())
			outBody.imagePoints[landmark]= vec3FromJson(imagePoints[landmark]);
		if (landmark < (int)visibility.size())
			outBody.visibility[landmark]= floatFromJson(visibility[landmark]);
		if (landmark < (int)worldPoints.size())
			outBody.worldPoints[landmark]= vec3FromJson(worldPoints[landmark]);
	}
}
} // namespace TrackingJson

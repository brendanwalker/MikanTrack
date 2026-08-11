#include "TestCommon.h"

// Writes (and opens) the printable origin aruco marker without launching the
// app. This is the extrinsics target: it defines the tracking world's origin
// AND its axes, so the sheet carries the FORWARD/LEFT/RIGHT labels that the
// fixed "right hand toward -Y" convention depends on.
//
// A charuco board was tried here instead and rejected: at the 60-70 cm working
// distance, the largest squares that fit on letter paper (24 mm) do not resolve
// enough corner detail to detect, while one large aruco square does.
static int runExportMarkerTool(const TestArgs& args)
{
	AppConfig config;
	config.load();

	int markerId= config.camera(0).extrinsics.markerId;
	float markerLengthMM= (float)config.camera(0).extrinsics.markerLengthMM;

	if (args.size() >= 2)
	{
		markerId= atoi(args[0].c_str());
		markerLengthMM= (float)atof(args[1].c_str());
	}
	else if (!args.empty())
	{
		MIKAN_LOG_ERROR("export-marker")
			<< "usage: --export-marker [markerId markerLengthMM]  (defaults to the saved config)";
		return 1;
	}

	MIKAN_LOG_INFO("export-marker") << "Marker: id " << markerId << ", " << markerLengthMM << " mm square";

	const std::filesystem::path exportPath=
		exportAndOpenArucoMarker(markerId, eCharucoDictionaryType::DICT_6X6, markerLengthMM);
	if (exportPath.empty())
	{
		MIKAN_LOG_ERROR("export-marker") << "Export failed";
		return 1;
	}

	MIKAN_LOG_INFO("export-marker") << "Wrote " << exportPath.string();
	return 0;
}

MIKAN_REGISTER_TEST("--export-marker", "Writes + opens the printable origin aruco marker PNG",
					eTestCategory::Tool, runExportMarkerTool);

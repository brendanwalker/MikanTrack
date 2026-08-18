#include "TestCommon.h"

#include "PathUtils.h"
#include "ProjectManager.h"

// Writes (and opens) the printable charuco board without launching the app.
// Uses the saved config's board parameters by default, so the sheet always
// matches what the wizards will detect; the optional arguments override them
// for a one-off print.
static int runExportBoardTool(const TestArgs& args)
{
	AppConfig config;
	ProjectManager::loadActiveProjectConfig(config);

	int cols= config.charucoBoard.cols;
	int rows= config.charucoBoard.rows;
	float squareLengthMM= (float)config.charucoBoard.squareLengthMM;
	float markerLengthMM= (float)config.charucoBoard.markerLengthMM;

	if (args.size() >= 4)
	{
		cols= atoi(args[0].c_str());
		rows= atoi(args[1].c_str());
		squareLengthMM= (float)atof(args[2].c_str());
		markerLengthMM= (float)atof(args[3].c_str());
	}
	else if (!args.empty())
	{
		MIKAN_LOG_ERROR("export-board")
			<< "usage: --export-board [cols rows squareMM markerMM]  (defaults to the saved config)";
		return 1;
	}

	MIKAN_LOG_INFO("export-board") << "Board: " << cols << "x" << rows << " squares, " << squareLengthMM
								   << " mm squares / " << markerLengthMM << " mm markers";

	const std::filesystem::path exportPath= exportAndOpenCharucoBoard(
		cols, rows, squareLengthMM, markerLengthMM, eCharucoDictionaryType::DICT_6X6);
	if (exportPath.empty())
	{
		MIKAN_LOG_ERROR("export-board") << "Export failed";
		return 1;
	}

	MIKAN_LOG_INFO("export-board") << "Wrote " << exportPath.string();
	return 0;
}

MIKAN_REGISTER_TEST("--export-board", "Writes + opens the printable charuco board PNG",
					eTestCategory::Tool, runExportBoardTool);

#pragma once

#include <array>
#include <string>

#include "glm/ext/matrix_double4x4.hpp"

#include "MikanVideoSourceTypes.h"

// Persisted application settings, stored as JSON at
// %APPDATA%/MikanMediaPipe/config.json

struct VideoConfig
{
	std::string deviceName;
	std::string devicePath;
	std::string modeName;
};

struct IntrinsicsConfig
{
	bool present= false;
	MikanMonoIntrinsics intrinsics;
	double reprojectionError= 0.0;
};

struct ExtrinsicsConfig
{
	bool present= false;
	glm::dmat4 markerFromCamera{1.0};
	int markerId= 0;
	double markerLengthMM= 100.0;
};

struct HandScaleConfig
{
	bool present= false;
	double refLengthMeters= 0.08; // wrist -> middle-MCP distance
};

struct CharucoBoardConfig
{
	int cols= 11;
	int rows= 8;
	double squareLengthMM= 16.0;
	double markerLengthMM= 12.0;
};

struct TrackingConfig
{
	bool flipHandedness= true;
	bool usePoseModel= true;
	bool poseHandSeededRoi= false; // overhead rigs: seed pose ROI from tracked hands
	int detectorIntervalFrames= 30;
	int poseFrameDivider= 2;
	float smoothingMinCutoff= 1.0f;
	float smoothingBeta= 0.05f;
	bool smoothingEnabled= true;
	std::string onnxEp= "directml"; // "directml" | "cpu"
};

struct OscConfig
{
	bool enabled= true;
	std::string targetIp= "127.0.0.1";
	int targetPort= 8000;
	int maxRateHz= 60;
};

class AppConfig
{
public:
	VideoConfig video;
	IntrinsicsConfig intrinsics;
	ExtrinsicsConfig extrinsics;
	HandScaleConfig handScale;
	CharucoBoardConfig charucoBoard;
	TrackingConfig tracking;
	OscConfig osc;

	// Loads from the default config path; returns false (and keeps defaults) if missing/corrupt
	bool load();
	bool save() const;

	void markDirty() { m_bDirty= true; }
	// Saves at most once per cooldown period when dirty; call from the main loop
	void updateAutoSave(float deltaSeconds);

	static std::string getConfigFilePath();

private:
	bool m_bDirty= false;
	float m_secondsSinceDirty= 0.f;
};

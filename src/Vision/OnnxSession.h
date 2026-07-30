#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

// Thin wrapper around Ort::Session with DirectML-first execution provider
// selection and cached input/output metadata.
//
// THREAD AFFINITY: All OnnxSession instances are expected to be created, run,
// and destroyed on the single inference (vision) thread. Ort::Session is not
// used concurrently anywhere in this app, and the shared Ort::Env outlives
// all sessions (it is a process-lifetime singleton).
class OnnxSession
{
public:
	OnnxSession()= default;
	~OnnxSession()= default;

	// Non-copyable
	OnnxSession(const OnnxSession&)= delete;
	OnnxSession& operator=(const OnnxSession&)= delete;

	// Loads the model at modelPath.
	// preferredEp: "directml" tries the DirectML EP (device 0) first with the
	// session options DML requires (mem pattern disabled, sequential
	// execution); on ANY failure it logs and falls back to a plain CPU
	// session. Anything else ("cpu", empty) goes straight to CPU.
	// Returns false if the model can't be loaded at all.
	bool create(const std::string& modelPath, const std::string& preferredEp);

	bool isValid() const { return m_session != nullptr; }

	// "DirectML" or "CPU" (or "none" before create succeeds)
	const char* activeEp() const { return m_activeEp.c_str(); }

	size_t getInputCount() const { return m_inputNames.size(); }
	size_t getOutputCount() const { return m_outputNames.size(); }
	const std::string& getInputName(size_t index) const { return m_inputNames[index]; }
	const std::string& getOutputName(size_t index) const { return m_outputNames[index]; }
	const std::vector<int64_t>& getInputShape(size_t index= 0) const { return m_inputShapes[index]; }
	const std::vector<int64_t>& getOutputShape(size_t index) const { return m_outputShapes[index]; }

	// Index of the first output whose last dimension equals lastDim, or -1.
	// Used to map detector outputs (scores vs box+keypoint regressors) by
	// shape rather than by graph order, since the opencv_zoo conversions
	// don't order them consistently across models.
	int findOutputByLastDim(int64_t lastDim) const;

	// Runs the session over all outputs. Inputs are in model input order.
	std::vector<Ort::Value> run(const Ort::Value* inputs, size_t inputCount);

	// CPU memory info for building input tensors over preallocated buffers
	static const Ort::MemoryInfo& getCpuMemoryInfo();

private:
	static Ort::Env& getSharedEnv();

	void cacheIoMetadata();
	void logModelInfo(const std::string& modelPath) const;

	std::unique_ptr<Ort::Session> m_session;
	std::string m_activeEp= "none";

	std::vector<std::string> m_inputNames;
	std::vector<std::string> m_outputNames;
	std::vector<const char*> m_inputNamePtrs;
	std::vector<const char*> m_outputNamePtrs;
	std::vector<std::vector<int64_t>> m_inputShapes;
	std::vector<std::vector<int64_t>> m_outputShapes;
};

p = "D:/Github/git-BrendanWalker/MikanMediaPipe/src/App/VisionThread.cpp"
s = open(p).read()
old = """	fusionConfig.stalenessWindowMs= m_config->fusion.stalenessWindowMs;
	fusionConfig.softmaxTemperature= m_config->fusion.softmaxTemperature;
	fusionConfig.wristMatchMaxDistM= m_config->fusion.wristMatchMaxDistM;"""
new = """	fusionConfig.stalenessWindowMs= m_config->fusion.stalenessWindowMs;
	fusionConfig.wristMatchMaxDistM= m_config->fusion.wristMatchMaxDistM;"""
assert old in s
s = s.replace(old, new)
open(p, "w").write(s)

p = "D:/Github/git-BrendanWalker/MikanMediaPipe/src/App/AppConfig.h"
s = open(p).read()
old = """	// A camera's last result older than this is excluded from fusion
	double stalenessWindowMs= 66.0;
	// Softmax sharpness for the per-landmark visibility weights: higher
	// values let the better view dominate faster
	float softmaxTemperature= 8.f;
	// Two cameras' world wrists further apart than this can't be the same"""
new = """	// A camera's last result older than this is excluded from fusion
	double stalenessWindowMs= 66.0;
	// Two cameras' world wrists further apart than this can't be the same"""
assert old in s
s = s.replace(old, new)
open(p, "w").write(s)

p = "D:/Github/git-BrendanWalker/MikanMediaPipe/src/App/AppConfig.cpp"
s = open(p).read()
old = """		fusion.stalenessWindowMs= fu.value("stalenessWindowMs", 66.0);
		fusion.softmaxTemperature= fu.value("softmaxTemperature", 8.f);
		fusion.wristMatchMaxDistM= fu.value("wristMatchMaxDistM", 0.25f);"""
new = """		fusion.stalenessWindowMs= fu.value("stalenessWindowMs", 66.0);
		fusion.wristMatchMaxDistM= fu.value("wristMatchMaxDistM", 0.25f);"""
assert old in s
s = s.replace(old, new)
old = """	j["fusion"]= {
		{"stalenessWindowMs", fusion.stalenessWindowMs},
		{"softmaxTemperature", fusion.softmaxTemperature},
		{"wristMatchMaxDistM", fusion.wristMatchMaxDistM},
	};"""
new = """	j["fusion"]= {
		{"stalenessWindowMs", fusion.stalenessWindowMs},
		{"wristMatchMaxDistM", fusion.wristMatchMaxDistM},
	};"""
assert old in s
s = s.replace(old, new)
open(p, "w").write(s)

p = "D:/Github/git-BrendanWalker/MikanMediaPipe/src/UI/SettingsPanels.cpp"
s = open(p).read()
old = """		ImGui::SetItemTooltip("A camera's last result older than this is\nexcluded from fusion");

		bChanged|= ImGui::SliderFloat("Softmax temperature", &fusion.softmaxTemperature, 1.f, 30.f, "%.1f");
		ImGui::SetItemTooltip("Higher = the better view dominates faster\n(30 approaches best-camera switching)");

		// Which camera won each hand in the last fusion"""
new = """		ImGui::SetItemTooltip("A camera's last result older than this is\nexcluded from fusion");

		// Which camera won each hand in the last fusion"""
assert old in s
s = s.replace(old, new)
open(p, "w").write(s)
print("config cleanup done")

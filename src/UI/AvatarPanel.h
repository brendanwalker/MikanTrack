#pragma once

#include <string>

class App;
class AppConfig;

// The avatar panel: which VRM is loaded, what the loader read from it, and
// the read-only humanoid bone map the retarget will drive. Loading goes
// through App (the model is main-thread, GL-adjacent state); this panel only
// edits the persisted AvatarConfig and asks App to load or clear.
class AvatarPanel
{
public:
	AvatarPanel(App* app, AppConfig* config);

	void draw(bool* pOpen);

private:
	void drawLoadControls();
	void drawModelSummary();
	void drawPlacement();
	void drawBoneTable();

	App* m_app;
	AppConfig* m_config;
};
